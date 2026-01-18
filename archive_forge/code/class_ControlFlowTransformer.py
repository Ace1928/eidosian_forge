import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.lang import directives
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import annos
from tensorflow.python.autograph.pyct.static_analysis import liveness
from tensorflow.python.autograph.pyct.static_analysis import reaching_definitions
from tensorflow.python.autograph.pyct.static_analysis import reaching_fndefs
class ControlFlowTransformer(converter.Base):
    """Transforms control flow structures like loops an conditionals."""

    def visit_Lambda(self, node):
        with self.state[_Function] as fn:
            fn.scope = anno.getanno(node, anno.Static.SCOPE)
            return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        with self.state[_Function] as fn:
            fn.scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)
            return self.generic_visit(node)

    def _create_nonlocal_declarations(self, vars_):
        vars_ = set(vars_)
        results = []
        global_vars = self.state[_Function].scope.globals & vars_
        if global_vars:
            results.append(gast.Global([str(v) for v in global_vars]))
        nonlocal_vars = [v for v in vars_ if not v.is_composite() and v not in global_vars]
        if nonlocal_vars:
            results.append(gast.Nonlocal([str(v) for v in nonlocal_vars]))
        return results

    def _create_state_functions(self, block_vars, nonlocal_declarations, getter_name, setter_name):
        if not block_vars:
            template = '\n        def getter_name():\n          return ()\n        def setter_name(block_vars):\n          pass\n      '
            return templates.replace(template, getter_name=getter_name, setter_name=setter_name)
        guarded_block_vars = []
        for v in block_vars:
            if v.is_simple():
                guarded_block_vars.append(v)
            else:
                guarded_block_vars.append(templates.replace_as_expression('ag__.ldu(lambda: var_, name)', var_=v, name=gast.Constant(str(v), kind=None)))
        template = '\n      def getter_name():\n        return guarded_state_vars,\n      def setter_name(vars_):\n        nonlocal_declarations\n        state_vars, = vars_\n    '
        return templates.replace(template, nonlocal_declarations=nonlocal_declarations, getter_name=getter_name, guarded_state_vars=guarded_block_vars, setter_name=setter_name, state_vars=tuple(block_vars))

    def _create_loop_options(self, node):
        if not anno.hasanno(node, anno.Basic.DIRECTIVES):
            return gast.Dict([], [])
        loop_directives = anno.getanno(node, anno.Basic.DIRECTIVES)
        if directives.set_loop_options not in loop_directives:
            return gast.Dict([], [])
        opts_dict = loop_directives[directives.set_loop_options]
        str_keys, values = zip(*opts_dict.items())
        keys = [gast.Constant(s, kind=None) for s in str_keys]
        values = list(values)
        return gast.Dict(keys, values)

    def _create_undefined_assigns(self, undefined_symbols):
        assignments = []
        for s in undefined_symbols:
            template = '\n        var = ag__.Undefined(symbol_name)\n      '
            assignments += templates.replace(template, var=s, symbol_name=gast.Constant(s.ssf(), kind=None))
        return assignments

    def _get_block_basic_vars(self, modified, live_in, live_out):
        nonlocals = self.state[_Function].scope.nonlocals
        basic_scope_vars = []
        for s in modified:
            if s.is_composite():
                continue
            if s in live_in or s in live_out or s in nonlocals:
                basic_scope_vars.append(s)
            continue
        return frozenset(basic_scope_vars)

    def _get_block_composite_vars(self, modified, live_in):
        composite_scope_vars = []
        for s in modified:
            if not s.is_composite():
                continue
            support_set_symbols = tuple((sss for sss in s.support_set if sss.is_symbol()))
            if not all((sss in live_in for sss in support_set_symbols)):
                continue
            composite_scope_vars.append(s)
        return frozenset(composite_scope_vars)

    def _get_block_vars(self, node, modified):
        """Determines the variables affected inside a control flow statement."""
        defined_in = anno.getanno(node, anno.Static.DEFINED_VARS_IN)
        live_in = anno.getanno(node, anno.Static.LIVE_VARS_IN)
        live_out = anno.getanno(node, anno.Static.LIVE_VARS_OUT)
        fn_scope = self.state[_Function].scope
        basic_scope_vars = self._get_block_basic_vars(modified, live_in, live_out)
        composite_scope_vars = self._get_block_composite_vars(modified, live_in)
        scope_vars = tuple(basic_scope_vars | composite_scope_vars)
        possibly_undefined = modified - defined_in - fn_scope.globals - fn_scope.nonlocals
        undefined = tuple((v for v in possibly_undefined if not v.is_composite()))
        input_only = basic_scope_vars & live_in - live_out
        scope_vars = sorted(scope_vars, key=lambda v: (v in input_only, v))
        nouts = len(scope_vars) - len(input_only)
        return (scope_vars, undefined, nouts)

    def visit_If(self, node):
        node = self.generic_visit(node)
        body_scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)
        orelse_scope = anno.getanno(node, annos.NodeAnno.ORELSE_SCOPE)
        cond_vars, undefined, nouts = self._get_block_vars(node, body_scope.bound | orelse_scope.bound)
        undefined_assigns = self._create_undefined_assigns(undefined)
        nonlocal_declarations = self._create_nonlocal_declarations(cond_vars)
        reserved = body_scope.referenced | orelse_scope.referenced
        state_getter_name = self.ctx.namer.new_symbol('get_state', reserved)
        state_setter_name = self.ctx.namer.new_symbol('set_state', reserved)
        state_functions = self._create_state_functions(cond_vars, nonlocal_declarations, state_getter_name, state_setter_name)
        orelse_body = node.orelse
        if not orelse_body:
            orelse_body = [gast.Pass()]
        template = '\n      state_functions\n      def body_name():\n        nonlocal_declarations\n        body\n      def orelse_name():\n        nonlocal_declarations\n        orelse\n      undefined_assigns\n      ag__.if_stmt(\n        test,\n        body_name,\n        orelse_name,\n        state_getter_name,\n        state_setter_name,\n        (symbol_names,),\n        nouts)\n    '
        new_nodes = templates.replace(template, body=node.body, body_name=self.ctx.namer.new_symbol('if_body', reserved), orelse=orelse_body, orelse_name=self.ctx.namer.new_symbol('else_body', reserved), nonlocal_declarations=nonlocal_declarations, nouts=gast.Constant(nouts, kind=None), state_functions=state_functions, state_getter_name=state_getter_name, state_setter_name=state_setter_name, symbol_names=tuple((gast.Constant(str(s), kind=None) for s in cond_vars)), test=node.test, undefined_assigns=undefined_assigns)
        origin_info.copy_origin(node, new_nodes[-1])
        return new_nodes

    def visit_While(self, node):
        node = self.generic_visit(node)
        body_scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)
        loop_vars, undefined, _ = self._get_block_vars(node, body_scope.bound)
        undefined_assigns = self._create_undefined_assigns(undefined)
        nonlocal_declarations = self._create_nonlocal_declarations(loop_vars)
        reserved = body_scope.referenced
        state_getter_name = self.ctx.namer.new_symbol('get_state', reserved)
        state_setter_name = self.ctx.namer.new_symbol('set_state', reserved)
        state_functions = self._create_state_functions(loop_vars, nonlocal_declarations, state_getter_name, state_setter_name)
        opts = self._create_loop_options(node)
        template = '\n      state_functions\n      def body_name():\n        nonlocal_declarations\n        body\n      def test_name():\n        return test\n      undefined_assigns\n      ag__.while_stmt(\n          test_name,\n          body_name,\n          state_getter_name,\n          state_setter_name,\n          (symbol_names,),\n          opts)\n    '
        new_nodes = templates.replace(template, body=node.body, body_name=self.ctx.namer.new_symbol('loop_body', reserved), nonlocal_declarations=nonlocal_declarations, opts=opts, state_functions=state_functions, state_getter_name=state_getter_name, state_setter_name=state_setter_name, symbol_names=tuple((gast.Constant(str(s), kind=None) for s in loop_vars)), test=node.test, test_name=self.ctx.namer.new_symbol('loop_test', reserved), undefined_assigns=undefined_assigns)
        origin_info.copy_origin(node, new_nodes[-1])
        return new_nodes

    def visit_For(self, node):
        node = self.generic_visit(node)
        body_scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)
        iter_scope = anno.getanno(node, annos.NodeAnno.ITERATE_SCOPE)
        loop_vars, undefined, _ = self._get_block_vars(node, body_scope.bound | iter_scope.bound)
        undefined_assigns = self._create_undefined_assigns(undefined)
        nonlocal_declarations = self._create_nonlocal_declarations(loop_vars)
        reserved = body_scope.referenced | iter_scope.referenced
        state_getter_name = self.ctx.namer.new_symbol('get_state', reserved)
        state_setter_name = self.ctx.namer.new_symbol('set_state', reserved)
        state_functions = self._create_state_functions(loop_vars, nonlocal_declarations, state_getter_name, state_setter_name)
        opts = self._create_loop_options(node)
        opts.keys.append(gast.Constant('iterate_names', kind=None))
        opts.values.append(gast.Constant(parser.unparse(node.target, include_encoding_marker=False), kind=None))
        if anno.hasanno(node, anno.Basic.EXTRA_LOOP_TEST):
            extra_test = anno.getanno(node, anno.Basic.EXTRA_LOOP_TEST)
            extra_test_name = self.ctx.namer.new_symbol('extra_test', reserved)
            template = '\n        def extra_test_name():\n          nonlocal_declarations\n          return extra_test_expr\n      '
            extra_test_function = templates.replace(template, extra_test_expr=extra_test, extra_test_name=extra_test_name, loop_vars=loop_vars, nonlocal_declarations=nonlocal_declarations)
        else:
            extra_test_name = parser.parse_expression('None')
            extra_test_function = []
        iterate_arg_name = self.ctx.namer.new_symbol('itr', reserved)
        template = '\n      iterates = iterate_arg_name\n    '
        iterate_expansion = templates.replace(template, iterate_arg_name=iterate_arg_name, iterates=node.target)
        origin_info.copy_origin(node, iterate_expansion)
        template = '\n      state_functions\n      def body_name(iterate_arg_name):\n        nonlocal_declarations\n        iterate_expansion\n        body\n      extra_test_function\n      undefined_assigns\n      ag__.for_stmt(\n          iterated,\n          extra_test_name,\n          body_name,\n          state_getter_name,\n          state_setter_name,\n          (symbol_names,),\n          opts)\n    '
        new_nodes = templates.replace(template, body=node.body, body_name=self.ctx.namer.new_symbol('loop_body', reserved), extra_test_function=extra_test_function, extra_test_name=extra_test_name, iterate_arg_name=iterate_arg_name, iterate_expansion=iterate_expansion, iterated=node.iter, nonlocal_declarations=nonlocal_declarations, opts=opts, symbol_names=tuple((gast.Constant(str(s), kind=None) for s in loop_vars)), state_functions=state_functions, state_getter_name=state_getter_name, state_setter_name=state_setter_name, undefined_assigns=undefined_assigns)
        origin_info.copy_origin(node, new_nodes[-1])
        return new_nodes