from pythran.passmanager import Transformation
from pythran.analyses import DefUseChains, UseDefChains, Identifiers
import gast as ast
class FalsePolymorphism(Transformation):
    """
    Rename variable when possible to avoid false polymorphism.

    >>> import gast as ast
    >>> from pythran import passmanager, backend
    >>> node = ast.parse("def foo(): a = 12; a = 'babar'")
    >>> pm = passmanager.PassManager("test")
    >>> _, node = pm.apply(FalsePolymorphism, node)
    >>> print(pm.dump(backend.Python, node))
    def foo():
        a = 12
        a_ = 'babar'
    """

    def __init__(self):
        super(FalsePolymorphism, self).__init__(DefUseChains, UseDefChains)

    def visit_FunctionDef(self, node):
        identifiers = self.gather(Identifiers, node)
        for def_ in self.def_use_chains.locals[node]:
            try:
                identifiers.remove(def_.name())
            except KeyError:
                pass
        visited_defs = set()
        for def_ in self.def_use_chains.locals[node]:
            if def_ in visited_defs:
                continue
            associated_defs = set()
            to_process = [def_]
            while to_process:
                curr = to_process.pop()
                if curr in associated_defs:
                    continue
                if curr.name() != def_.name():
                    continue
                associated_defs.add(curr)
                for u in curr.users():
                    to_process.append(u)
                curr_udc = (d for d in self.use_def_chains.get(curr.node, []) if isinstance(d.node, ast.Name))
                to_process.extend(curr_udc)
            visited_defs.update(associated_defs)
            local_identifier = def_.name()
            name = local_identifier
            while name in identifiers:
                name += '_'
            identifiers.add(name)
            if name == local_identifier:
                continue
            self.update = True
            for d in associated_defs:
                dn = d.node
                if isinstance(dn, ast.Name) and dn.id == local_identifier:
                    dn.id = name
        return node