import contextlib
import imp
import inspect
import io
import sys
from tensorflow.python.autograph.core import config
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.impl import api
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
class TestingTranspiler(api.PyToTF):
    """Testing version that only applies given transformations."""

    def __init__(self, converters, ag_overrides):
        super(TestingTranspiler, self).__init__()
        if isinstance(converters, (list, tuple)):
            self._converters = converters
        else:
            self._converters = (converters,)
        self.transformed_ast = None
        self._ag_overrides = ag_overrides

    def get_extra_locals(self):
        retval = super(TestingTranspiler, self).get_extra_locals()
        if self._ag_overrides:
            modified_ag = imp.new_module('fake_autograph')
            modified_ag.__dict__.update(retval['ag__'].__dict__)
            modified_ag.__dict__.update(self._ag_overrides)
            retval['ag__'] = modified_ag
        return retval

    def transform_ast(self, node, ctx):
        node = self.initial_analysis(node, ctx)
        for c in self._converters:
            node = c.transform(node, ctx)
        self.transformed_ast = node
        self.transform_ctx = ctx
        return node