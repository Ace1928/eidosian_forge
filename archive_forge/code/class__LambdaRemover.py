from pythran.analyses import GlobalDeclarations, ImportedIds
from pythran.analyses import Check
from pythran.analyses import ExtendedDefUseChains
from pythran.passmanager import Transformation
from pythran.tables import MODULES
from pythran.conversion import mangle
import pythran.metadata as metadata
from copy import copy, deepcopy
import gast as ast
class _LambdaRemover(ast.NodeTransformer):

    def __init__(self, parent, prefix):
        super(_LambdaRemover, self).__init__()
        self.prefix = prefix
        self.parent = parent
        self.patterns = parent.patterns

    def __getattr__(self, attr):
        return getattr(self.parent, attr)

    def visit_Lambda(self, node):
        op = issimpleoperator(node)
        if op is not None:
            if mangle('operator') not in self.global_declarations:
                import_ = ast.Import([ast.alias('operator', mangle('operator'))])
                self.imports.append(import_)
                operator_module = MODULES['operator']
                self.global_declarations[mangle('operator')] = operator_module
            return ast.Attribute(ast.Name(mangle('operator'), ast.Load(), None, None), op, ast.Load())
        self.generic_visit(node)
        forged_name = '{0}_lambda{1}'.format(self.prefix, len(self.lambda_functions))
        ii = self.gather(ImportedIds, node)
        ii.difference_update(self.lambda_functions)
        binded_args = [ast.Name(iin, ast.Load(), None, None) for iin in sorted(ii)]
        node.args.args = [ast.Name(iin, ast.Param(), None, None) for iin in sorted(ii)] + node.args.args
        for patternname, pattern in self.patterns.items():
            if issamelambda(pattern, node):
                proxy_call = ast.Name(patternname, ast.Load(), None, None)
                break
        else:
            duc = ExtendedDefUseChains()
            nodepattern = deepcopy(node)
            duc.visit(ast.Module([ast.Expr(nodepattern)], []))
            self.patterns[forged_name] = (nodepattern, duc)
            forged_fdef = ast.FunctionDef(forged_name, copy(node.args), [ast.Return(node.body)], [], None, None)
            metadata.add(forged_fdef, metadata.Local())
            self.lambda_functions.append(forged_fdef)
            self.global_declarations[forged_name] = forged_fdef
            proxy_call = ast.Name(forged_name, ast.Load(), None, None)
        if binded_args:
            if MODULES['functools'] not in self.global_declarations.values():
                import_ = ast.Import([ast.alias('functools', mangle('functools'))])
                self.imports.append(import_)
                functools_module = MODULES['functools']
                self.global_declarations[mangle('functools')] = functools_module
            return ast.Call(ast.Attribute(ast.Name(mangle('functools'), ast.Load(), None, None), 'partial', ast.Load()), [proxy_call] + binded_args, [])
        else:
            return proxy_call