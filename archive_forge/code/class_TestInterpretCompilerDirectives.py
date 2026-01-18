import os.path
import unittest
from Cython.TestUtils import TransformTest
from Cython.Compiler.ParseTreeTransforms import *
from Cython.Compiler.ParseTreeTransforms import _calculate_pickle_checksums
from Cython.Compiler.Nodes import *
from Cython.Compiler import Main, Symtab, Options
class TestInterpretCompilerDirectives(TransformTest):
    """
    This class tests the parallel directives AST-rewriting and importing.
    """
    import_code = u'\n        cimport cython.parallel\n        cimport cython.parallel as par\n        from cython cimport parallel as par2\n        from cython cimport parallel\n\n        from cython.parallel cimport threadid as tid\n        from cython.parallel cimport threadavailable as tavail\n        from cython.parallel cimport prange\n    '
    expected_directives_dict = {u'cython.parallel': u'cython.parallel', u'par': u'cython.parallel', u'par2': u'cython.parallel', u'parallel': u'cython.parallel', u'tid': u'cython.parallel.threadid', u'tavail': u'cython.parallel.threadavailable', u'prange': u'cython.parallel.prange'}

    def setUp(self):
        super(TestInterpretCompilerDirectives, self).setUp()
        compilation_options = Options.CompilationOptions(Options.default_options)
        ctx = Main.Context.from_options(compilation_options)
        transform = InterpretCompilerDirectives(ctx, ctx.compiler_directives)
        transform.module_scope = Symtab.ModuleScope('__main__', None, ctx)
        self.pipeline = [transform]
        self.debug_exception_on_error = DebugFlags.debug_exception_on_error

    def tearDown(self):
        DebugFlags.debug_exception_on_error = self.debug_exception_on_error

    def test_parallel_directives_cimports(self):
        self.run_pipeline(self.pipeline, self.import_code)
        parallel_directives = self.pipeline[0].parallel_directives
        self.assertEqual(parallel_directives, self.expected_directives_dict)

    def test_parallel_directives_imports(self):
        self.run_pipeline(self.pipeline, self.import_code.replace(u'cimport', u'import'))
        parallel_directives = self.pipeline[0].parallel_directives
        self.assertEqual(parallel_directives, self.expected_directives_dict)