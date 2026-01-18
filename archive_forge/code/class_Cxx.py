from pythran.analyses import LocalNodeDeclarations, GlobalDeclarations, Scope
from pythran.analyses import YieldPoints, IsAssigned, ASTMatcher, AST_any
from pythran.analyses import RangeValues, PureExpressions, Dependencies
from pythran.analyses import Immediates, Ancestors, StrictAliases
from pythran.config import cfg
from pythran.cxxgen import Template, Include, Namespace, CompilationUnit
from pythran.cxxgen import Statement, Block, AnnotatedStatement, Typedef, Label
from pythran.cxxgen import Value, FunctionDeclaration, EmptyStatement, Nop
from pythran.cxxgen import FunctionBody, Line, ReturnStatement, Struct, Assign
from pythran.cxxgen import For, While, TryExcept, ExceptHandler, If, AutoFor
from pythran.cxxgen import StatementWithComments
from pythran.openmp import OMPDirective
from pythran.passmanager import Backend
from pythran.syntax import PythranSyntaxError
from pythran.tables import operator_to_lambda, update_operator_to_lambda
from pythran.tables import pythran_ward, attributes as attributes_table
from pythran.types.conversion import PYTYPE_TO_CTYPE_TABLE, TYPE_TO_SUFFIX
from pythran.types.types import Types
from pythran.utils import attr_to_path, pushpop, cxxid, isstr, isnum
from pythran.utils import isextslice, ispowi, quote_cxxstring
from pythran import metadata, unparse
from math import isnan, isinf
import gast as ast
from functools import reduce
import io
class Cxx(Backend):
    """
    Produces a C++ representation of the AST.

    >>> import gast as ast, pythran.passmanager as passmanager, os
    >>> node = ast.parse("def foo(): return 'hello world'")
    >>> pm = passmanager.PassManager('test')
    >>> r = pm.dump(Cxx, node)
    >>> print(str(r).replace(os.sep, '/'))
    #include <pythonic/include/types/str.hpp>
    #include <pythonic/types/str.hpp>
    namespace 
    {
      namespace __pythran_test
      {
        struct foo
        {
          typedef void callable;
          typedef void pure;
          struct type
          {
            typedef pythonic::types::str __type0;
            typedef typename pythonic::returnable<__type0>::type __type1;
            typedef __type1 result_type;
          }  ;
          inline
          typename type::result_type operator()() const;
          ;
        }  ;
        inline
        typename foo::type::result_type foo::operator()() const
        {
          return pythonic::types::str("hello world");
        }
      }
    }
    """

    def __init__(self):
        """ Basic initialiser gathering analysis informations. """
        self.result = None
        super(Cxx, self).__init__(Dependencies, GlobalDeclarations, Types, Scope, RangeValues, PureExpressions, Immediates, Ancestors, StrictAliases)

    def visit_Module(self, node):
        """ Build a compilation unit. """
        if cfg.getboolean('backend', 'annotate'):
            node = ast.fix_missing_locations(node)
        header_deps = sorted(self.dependencies)
        headers = [Include('/'.join(['pythonic', 'include'] + [cxxid(x) for x in t]) + '.hpp') for t in header_deps]
        headers += [Include('/'.join(['pythonic'] + [cxxid(x) for x in t]) + '.hpp') for t in header_deps]
        decls_n_defns = list(filter(None, (self.visit(stmt) for stmt in node.body)))
        decls, defns = zip(*decls_n_defns) if decls_n_defns else ([], [])
        nsbody = [s for ls in decls + defns for s in ls]
        ns = Namespace(pythran_ward + self.passmanager.module_name, nsbody)
        anonymous_ns = Namespace('', [ns])
        self.result = CompilationUnit(headers + [anonymous_ns])

    def visit_FunctionDef(self, node):
        yields = self.gather(YieldPoints, node)
        visitor = (CxxGenerator if yields else CxxFunction)(self)
        return visitor.visit(node)