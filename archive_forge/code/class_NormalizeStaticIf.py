from pythran.analyses import (ImportedIds, HasReturn, IsAssigned, CFG,
from pythran.passmanager import Transformation
from pythran.syntax import PythranSyntaxError
import gast as ast
from copy import deepcopy
class NormalizeStaticIf(Transformation):

    def __init__(self):
        super(NormalizeStaticIf, self).__init__(StaticExpressions, Ancestors, DefUseChains)

    def visit_Module(self, node):
        self.new_functions = []
        self.funcs = []
        self.cfgs = []
        self.generic_visit(node)
        node.body.extend(self.new_functions)
        return node

    def escaping_ids(self, scope_stmt, stmts):
        """gather sets of identifiers defined in stmts and used out of it"""
        assigned_nodes = self.gather(IsAssigned, self.make_fake(stmts))
        escaping = set()
        for assigned_node in assigned_nodes:
            head = self.def_use_chains.chains[assigned_node]
            for user in head.users():
                if scope_stmt not in self.ancestors[user.node]:
                    escaping.add(head.name())
        return escaping

    @staticmethod
    def make_fake(stmts):
        return ast.If(ast.Constant(0, None), stmts, [])

    @staticmethod
    def make_dispatcher(static_expr, func_true, func_false, imported_ids):
        dispatcher_args = [static_expr, ast.Name(func_true.name, ast.Load(), None, None), ast.Name(func_false.name, ast.Load(), None, None)]
        dispatcher = ast.Call(ast.Attribute(ast.Attribute(ast.Name('builtins', ast.Load(), None, None), 'pythran', ast.Load()), 'static_if', ast.Load()), dispatcher_args, [])
        actual_call = ast.Call(dispatcher, [ast.Name(ii, ast.Load(), None, None) for ii in imported_ids], [])
        return actual_call

    def true_name(self):
        return '$isstatic{}'.format(len(self.new_functions) + 0)

    def false_name(self):
        return '$isstatic{}'.format(len(self.new_functions) + 1)

    def visit_FunctionDef(self, node):
        self.cfgs.append(self.gather(CFG, node))
        self.funcs.append(node)
        onode = self.generic_visit(node)
        self.funcs.pop()
        self.cfgs.pop()
        return onode

    def visit_IfExp(self, node):
        self.generic_visit(node)
        if node.test not in self.static_expressions:
            return node
        imported_ids = sorted(self.gather(ImportedIds, node))
        func_true = outline(self.true_name(), imported_ids, [], node.body, False, False, False)
        func_false = outline(self.false_name(), imported_ids, [], node.orelse, False, False, False)
        self.new_functions.extend((func_true, func_false))
        actual_call = self.make_dispatcher(node.test, func_true, func_false, imported_ids)
        return actual_call

    def make_control_flow_handlers(self, cont_n, status_n, expected_return, has_cont, has_break):
        """
        Create the statements in charge of gathering control flow information
        for the static_if result, and executes the expected control flow
        instruction
        """
        if expected_return:
            assign = cont_ass = [ast.Assign([ast.Tuple(expected_return, ast.Store())], ast.Name(cont_n, ast.Load(), None, None), None)]
        else:
            assign = cont_ass = []
        if has_cont:
            cmpr = ast.Compare(ast.Name(status_n, ast.Load(), None, None), [ast.Eq()], [ast.Constant(LOOP_CONT, None)])
            cont_ass = [ast.If(cmpr, deepcopy(assign) + [ast.Continue()], cont_ass)]
        if has_break:
            cmpr = ast.Compare(ast.Name(status_n, ast.Load(), None, None), [ast.Eq()], [ast.Constant(LOOP_BREAK, None)])
            cont_ass = [ast.If(cmpr, deepcopy(assign) + [ast.Break()], cont_ass)]
        return cont_ass

    def visit_If(self, node):
        if node.test not in self.static_expressions:
            return self.generic_visit(node)
        imported_ids = self.gather(ImportedIds, node)
        assigned_ids_left = self.escaping_ids(node, node.body)
        assigned_ids_right = self.escaping_ids(node, node.orelse)
        assigned_ids_both = assigned_ids_left.union(assigned_ids_right)
        imported_ids.update((i for i in assigned_ids_left if i not in assigned_ids_right))
        imported_ids.update((i for i in assigned_ids_right if i not in assigned_ids_left))
        imported_ids = sorted(imported_ids)
        assigned_ids = sorted(assigned_ids_both)
        fbody = self.make_fake(node.body)
        true_has_return = self.gather(HasReturn, fbody)
        true_has_break = self.gather(HasBreak, fbody)
        true_has_cont = self.gather(HasContinue, fbody)
        felse = self.make_fake(node.orelse)
        false_has_return = self.gather(HasReturn, felse)
        false_has_break = self.gather(HasBreak, felse)
        false_has_cont = self.gather(HasContinue, felse)
        has_return = true_has_return or false_has_return
        has_break = true_has_break or false_has_break
        has_cont = true_has_cont or false_has_cont
        self.generic_visit(node)
        func_true = outline(self.true_name(), imported_ids, assigned_ids, node.body, has_return, has_break, has_cont)
        func_false = outline(self.false_name(), imported_ids, assigned_ids, node.orelse, has_return, has_break, has_cont)
        self.new_functions.extend((func_true, func_false))
        actual_call = self.make_dispatcher(node.test, func_true, func_false, imported_ids)
        expected_return = [ast.Name(ii, ast.Store(), None, None) for ii in assigned_ids]
        self.update = True
        n = len(self.new_functions)
        status_n = '$status{}'.format(n)
        return_n = '$return{}'.format(n)
        cont_n = '$cont{}'.format(n)
        if has_return:
            cfg = self.cfgs[-1]
            always_return = all((isinstance(x, (ast.Return, ast.Yield)) for x in cfg[node]))
            always_return &= true_has_return and false_has_return
            fast_return = [ast.Name(status_n, ast.Store(), None, None), ast.Name(return_n, ast.Store(), None, None), ast.Name(cont_n, ast.Store(), None, None)]
            if always_return:
                return [ast.Assign([ast.Tuple(fast_return, ast.Store())], actual_call, None), ast.Return(ast.Name(return_n, ast.Load(), None, None))]
            else:
                cont_ass = self.make_control_flow_handlers(cont_n, status_n, expected_return, has_cont, has_break)
                cmpr = ast.Compare(ast.Name(status_n, ast.Load(), None, None), [ast.Eq()], [ast.Constant(EARLY_RET, None)])
                return [ast.Assign([ast.Tuple(fast_return, ast.Store())], actual_call, None), ast.If(cmpr, [ast.Return(ast.Name(return_n, ast.Load(), None, None))], cont_ass)]
        elif has_break or has_cont:
            cont_ass = self.make_control_flow_handlers(cont_n, status_n, expected_return, has_cont, has_break)
            fast_return = [ast.Name(status_n, ast.Store(), None, None), ast.Name(cont_n, ast.Store(), None, None)]
            return [ast.Assign([ast.Tuple(fast_return, ast.Store())], actual_call, None)] + cont_ass
        elif expected_return:
            return ast.Assign([ast.Tuple(expected_return, ast.Store())], actual_call, None)
        else:
            return ast.Expr(actual_call)