import inspect
import logging
import sys
from copy import deepcopy
from collections import deque
from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.common.errors import DeveloperError, TemplateExpressionError
from pyomo.common.numeric_types import (
import pyomo.core.expr.expr_common as common
from pyomo.core.expr.symbol_map import SymbolMap
class StreamBasedExpressionVisitor(object):
    """This class implements a generic stream-based expression walker.

    This visitor walks an expression tree using a depth-first strategy
    and generates a full event stream similar to other tree visitors
    (e.g., the expat XML parser).  The following events are triggered
    through callback functions as the traversal enters and leaves nodes
    in the tree:

    ::

       initializeWalker(expr) -> walk, result
       enterNode(N1) -> args, data
       {for N2 in args:}
         beforeChild(N1, N2) -> descend, child_result
           enterNode(N2) -> N2_args, N2_data
           [...]
           exitNode(N2, n2_data) -> child_result
         acceptChildResult(N1, data, child_result) -> data
         afterChild(N1, N2) -> None
       exitNode(N1, data) -> N1_result
       finalizeWalker(result) -> result

    Individual event callbacks match the following signatures:

    walk, result = initializeWalker(self, expr):

         initializeWalker() is called to set the walker up and perform
         any preliminary processing on the root node.  The method returns
         a flag indicating if the tree should be walked and a result.  If
         `walk` is True, then result is ignored.  If `walk` is False,
         then `result` is returned as the final result from the walker,
         bypassing all other callbacks (including finalizeResult).

    args, data = enterNode(self, node):

         enterNode() is called when the walker first enters a node (from
         above), and is passed the node being entered.  It is expected to
         return a tuple of child `args` (as either a tuple or list) and a
         user-specified data structure for collecting results.  If None
         is returned for args, the node's args attribute is used for
         expression types and the empty tuple for leaf nodes.  Returning
         None is equivalent to returning (None,None).  If the callback is
         not defined, the default behavior is equivalent to returning
         (None, []).

    node_result = exitNode(self, node, data):

         exitNode() is called after the node is completely processed (as
         the walker returns up the tree to the parent node).  It is
         passed the node and the results data structure (defined by
         enterNode() and possibly further modified by
         acceptChildResult()), and is expected to return the "result" for
         this node.  If not specified, the default action is to return
         the data object from enterNode().

    descend, child_result = beforeChild(self, node, child, child_idx):

         beforeChild() is called by a node for every child before
         entering the child node.  The node, child node, and child index
         (position in the args list from enterNode()) are passed as
         arguments.  beforeChild should return a tuple (descend,
         child_result).  If descend is False, the child node will not be
         entered and the value returned to child_result will be passed to
         the node's acceptChildResult callback.  Returning None is
         equivalent to (True, None).  The default behavior if not
         specified is equivalent to (True, None).

    data = acceptChildResult(self, node, data, child_result, child_idx):

         acceptChildResult() is called for each child result being
         returned to a node.  This callback is responsible for recording
         the result for later processing or passing up the tree.  It is
         passed the node, result data structure (see enterNode()), child
         result, and the child index (position in args from enterNode()).
         The data structure (possibly modified or replaced) must be
         returned.  If acceptChildResult is not specified, it does
         nothing if data is None, otherwise it calls data.append(result).

    afterChild(self, node, child, child_idx):

         afterChild() is called by a node for every child node
         immediately after processing the node is complete before control
         moves to the next child or up to the parent node.  The node,
         child node, an child index (position in args from enterNode())
         are passed, and nothing is returned.  If afterChild is not
         specified, no action takes place.

    finalizeResult(self, result):

         finalizeResult() is called once after the entire expression tree
         has been walked.  It is passed the result returned by the root
         node exitNode() callback.  If finalizeResult is not specified,
         the walker returns the result obtained from the exitNode
         callback on the root node.

    Clients interact with this class by either deriving from it and
    implementing the necessary callbacks (see above), assigning callable
    functions to an instance of this class, or passing the callback
    functions as arguments to this class' constructor.

    """
    client_methods = {'enterNode': 'e', 'exitNode': 'x', 'beforeChild': 'b', 'afterChild': 'a', 'acceptChildResult': 'c', 'initializeWalker': '', 'finalizeResult': ''}

    def __init__(self, **kwds):
        for field in self.client_methods:
            if field in kwds:
                setattr(self, field, kwds.pop(field))
            elif not hasattr(self, field):
                setattr(self, field, None)
        if kwds:
            raise RuntimeError('Unrecognized keyword arguments: %s' % (kwds,))
        _fcns = (('beforeChild', 2), ('acceptChildResult', 3), ('afterChild', 2))
        for name, nargs in _fcns:
            fcn = getattr(self, name)
            if fcn is None:
                continue
            _args = inspect.getfullargspec(fcn)
            _self_arg = 1 if inspect.ismethod(fcn) else 0
            if len(_args.args) == nargs + _self_arg and _args.varargs is None:
                deprecation_warning('Note that the API for the StreamBasedExpressionVisitor has changed to include the child index for the %s() method.  Please update your walker callbacks.' % (name,), version='5.7.0')

                def wrap(fcn, nargs):

                    def wrapper(*args):
                        return fcn(*args[:nargs])
                    return wrapper
                setattr(self, name, wrap(fcn, nargs))
        self.recursion_stack = None
        recursive_node_handler = '_process_node_' + ''.join(sorted(('' if getattr(self, f[0]) is None else f[1] for f in self.client_methods.items())))
        self._process_node = getattr(self, recursive_node_handler, self._process_node_general)

    def walk_expression(self, expr):
        """Walk an expression, calling registered callbacks.

        This is the standard interface for running the visitor.  It
        defaults to using an efficient recursive implementation of the
        visitor, falling back on :py:meth:`walk_expression_nonrecursive`
        if the recursion stack gets too deep.

        """
        if self.initializeWalker is not None:
            walk, root = self.initializeWalker(expr)
            if not walk:
                return root
            elif root is None:
                root = expr
        else:
            root = expr
        try:
            result = self._process_node(root, RECURSION_LIMIT)
            _nonrecursive = None
        except RevertToNonrecursive:
            ptr = (None,) + self.recursion_stack.pop()
            while self.recursion_stack:
                ptr = (ptr,) + self.recursion_stack.pop()
            self.recursion_stack = None
            _nonrecursive = (self._nonrecursive_walker_loop, ptr)
        except RecursionError:
            logger.warning('Unexpected RecursionError walking an expression tree.', extra={'id': 'W1003'})
            _nonrecursive = (self.walk_expression_nonrecursive, expr)
        if _nonrecursive is not None:
            return _nonrecursive[0](_nonrecursive[1])
        if self.finalizeResult is not None:
            return self.finalizeResult(result)
        else:
            return result

    def _compute_actual_recursion_limit(self):
        recursion_limit = sys.getrecursionlimit() - get_stack_depth() - 2 * RECURSION_LIMIT
        if recursion_limit <= RECURSION_LIMIT:
            self.recursion_stack = []
            raise RevertToNonrecursive()
        return recursion_limit

    def _process_node_general(self, node, recursion_limit):
        """Recursive routine for processing nodes with general callbacks

        This is the "general" implementation of the
        StreamBasedExpressionVisitor node processor that can handle any
        combination of registered callback functions.

        """
        if not recursion_limit:
            recursion_limit = self._compute_actual_recursion_limit()
        else:
            recursion_limit -= 1
        if self.enterNode is not None:
            tmp = self.enterNode(node)
            if tmp is None:
                args = data = None
            else:
                args, data = tmp
        else:
            args = None
            data = []
        if args is None:
            if type(node) in nonpyomo_leaf_types or not node.is_expression_type():
                args = ()
            else:
                args = node.args
        context_manager = hasattr(args, '__enter__')
        if context_manager:
            args.__enter__()
        try:
            descend = True
            child_idx = -1
            arg_iter = iter(args)
            for child in arg_iter:
                child_idx += 1
                if self.beforeChild is not None:
                    tmp = self.beforeChild(node, child, child_idx)
                    if tmp is None:
                        descend = True
                    else:
                        descend, child_result = tmp
                if descend:
                    child_result = self._process_node(child, recursion_limit)
                if self.acceptChildResult is not None:
                    data = self.acceptChildResult(node, data, child_result, child_idx)
                elif data is not None:
                    data.append(child_result)
                if self.afterChild is not None:
                    self.afterChild(node, child, child_idx)
        except RevertToNonrecursive:
            self._recursive_frame_to_nonrecursive_stack(locals())
            context_manager = False
            raise
        finally:
            if context_manager:
                args.__exit__(None, None, None)
        if self.exitNode is not None:
            return self.exitNode(node, data)
        else:
            return data

    def _process_node_bex(self, node, recursion_limit):
        """Recursive routine for processing nodes with only 'bex' callbacks

        This is a special-case implementation of the "general"
        StreamBasedExpressionVisitor node processor for the case that
        only beforeChild, enterNode, and exitNode are defined (see
        also the definition of the client_methods dict).

        """
        if not recursion_limit:
            recursion_limit = self._compute_actual_recursion_limit()
        else:
            recursion_limit -= 1
        tmp = self.enterNode(node)
        if tmp is None:
            args = data = None
        else:
            args, data = tmp
        if args is None:
            if type(node) in nonpyomo_leaf_types or not node.is_expression_type():
                args = ()
            else:
                args = node.args
        context_manager = hasattr(args, '__enter__')
        if context_manager:
            args.__enter__()
        try:
            child_idx = -1
            arg_iter = iter(args)
            for child in arg_iter:
                child_idx += 1
                tmp = self.beforeChild(node, child, child_idx)
                if tmp is None:
                    descend = True
                else:
                    descend, child_result = tmp
                if descend:
                    data.append(self._process_node(child, recursion_limit))
                else:
                    data.append(child_result)
        except RevertToNonrecursive:
            self._recursive_frame_to_nonrecursive_stack(locals())
            context_manager = False
            raise
        finally:
            if context_manager:
                args.__exit__(None, None, None)
        return self.exitNode(node, data)

    def _process_node_bx(self, node, recursion_limit):
        """Recursive routine for processing nodes with only 'bx' callbacks

        This is a special-case implementation of the "general"
        StreamBasedExpressionVisitor node processor for the case that
        only beforeChild and exitNode are defined (see also the
        definition of the client_methods dict).

        """
        if not recursion_limit:
            recursion_limit = self._compute_actual_recursion_limit()
        else:
            recursion_limit -= 1
        if type(node) in nonpyomo_leaf_types or not node.is_expression_type():
            args = ()
        else:
            args = node.args
        data = []
        try:
            child_idx = -1
            arg_iter = iter(args)
            for child in arg_iter:
                child_idx += 1
                tmp = self.beforeChild(node, child, child_idx)
                if tmp is None:
                    descend = True
                else:
                    descend, child_result = tmp
                if descend:
                    data.append(self._process_node(child, recursion_limit))
                else:
                    data.append(child_result)
        except RevertToNonrecursive:
            self._recursive_frame_to_nonrecursive_stack(locals())
            raise
        finally:
            pass
        return self.exitNode(node, data)

    def _recursive_frame_to_nonrecursive_stack(self, local):
        child_idx = local['child_idx']
        _arg_list = [None] * child_idx
        _arg_list.append(local['child'])
        _arg_list.extend(local['arg_iter'])
        if not self.recursion_stack:
            child_idx -= 1
        self.recursion_stack.append((local['node'], _arg_list, len(_arg_list) - 1, local['data'], child_idx))

    def walk_expression_nonrecursive(self, expr):
        """Nonrecursively walk an expression, calling registered callbacks.

        This routine is safer than the recursive walkers for deep (or
        unbalanced) trees.  It is, however, slightly slower than the
        recursive implementations.

        """
        if self.initializeWalker is not None:
            walk, result = self.initializeWalker(expr)
            if not walk:
                return result
            elif result is not None:
                expr = result
        if self.enterNode is not None:
            tmp = self.enterNode(expr)
            if tmp is None:
                args = data = None
            else:
                args, data = tmp
        else:
            args = None
            data = []
        if args is None:
            if type(expr) in nonpyomo_leaf_types or not expr.is_expression_type():
                args = ()
            else:
                args = expr.args
        if hasattr(args, '__enter__'):
            args.__enter__()
        node = expr
        return self._nonrecursive_walker_loop((None, node, args, len(args) - 1, data, -1))

    def _nonrecursive_walker_loop(self, ptr):
        _, node, args, _, data, child_idx = ptr
        try:
            while 1:
                if child_idx < ptr[3]:
                    child_idx += 1
                    child = ptr[2][child_idx]
                    if self.beforeChild is not None:
                        tmp = self.beforeChild(node, child, child_idx)
                        if tmp is None:
                            descend = True
                            child_result = None
                        else:
                            descend, child_result = tmp
                        if not descend:
                            if self.acceptChildResult is not None:
                                data = self.acceptChildResult(node, data, child_result, child_idx)
                            elif data is not None:
                                data.append(child_result)
                            if self.afterChild is not None:
                                self.afterChild(node, child, child_idx)
                            continue
                    ptr = ptr[:4] + (data, child_idx)
                    if self.enterNode is not None:
                        tmp = self.enterNode(child)
                        if tmp is None:
                            args = data = None
                        else:
                            args, data = tmp
                    else:
                        args = None
                        data = []
                    if args is None:
                        if type(child) in nonpyomo_leaf_types or not child.is_expression_type():
                            args = ()
                        else:
                            args = child.args
                    if hasattr(args, '__enter__'):
                        args.__enter__()
                    node = child
                    child_idx = -1
                    ptr = (ptr, node, args, len(args) - 1, data, child_idx)
                else:
                    if hasattr(ptr[2], '__exit__'):
                        ptr[2].__exit__(None, None, None)
                    if self.exitNode is not None:
                        node_result = self.exitNode(node, data)
                    else:
                        node_result = data
                    ptr = ptr[0]
                    if ptr is None:
                        if self.finalizeResult is not None:
                            return self.finalizeResult(node_result)
                        else:
                            return node_result
                    node, child = (ptr[1], node)
                    data = ptr[4]
                    child_idx = ptr[5]
                    if self.acceptChildResult is not None:
                        data = self.acceptChildResult(node, data, node_result, child_idx)
                    elif data is not None:
                        data.append(node_result)
                    if self.afterChild is not None:
                        self.afterChild(node, child, child_idx)
        finally:
            while ptr is not None:
                if hasattr(ptr[2], '__exit__'):
                    ptr[2].__exit__(None, None, None)
                ptr = ptr[0]