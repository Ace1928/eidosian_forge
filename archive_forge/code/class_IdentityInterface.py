import os
import re
import numpy as np
from ..base import (
from ..io import IOBase, add_traits
from ...utils.filemanip import ensure_list, copyfile, split_filename
class IdentityInterface(IOBase):
    """Basic interface class generates identity mappings

    Examples
    --------

    >>> from nipype.interfaces.utility import IdentityInterface
    >>> ii = IdentityInterface(fields=['a', 'b'], mandatory_inputs=False)
    >>> ii.inputs.a
    <undefined>

    >>> ii.inputs.a = 'foo'
    >>> out = ii._outputs()
    >>> out.a
    <undefined>

    >>> out = ii.run()
    >>> out.outputs.a
    'foo'

    >>> ii2 = IdentityInterface(fields=['a', 'b'], mandatory_inputs=True)
    >>> ii2.inputs.a = 'foo'
    >>> out = ii2.run() # doctest: +SKIP
    ValueError: IdentityInterface requires a value for input 'b' because it was listed in 'fields' Interface IdentityInterface failed to run.
    """
    input_spec = DynamicTraitedSpec
    output_spec = DynamicTraitedSpec

    def __init__(self, fields=None, mandatory_inputs=True, **inputs):
        super(IdentityInterface, self).__init__(**inputs)
        if fields is None or not fields:
            raise ValueError('Identity Interface fields must be a non-empty list')
        for in_field in inputs:
            if in_field not in fields:
                raise ValueError('Identity Interface input is not in the fields: %s' % in_field)
        self._fields = fields
        self._mandatory_inputs = mandatory_inputs
        add_traits(self.inputs, fields)
        self.inputs.trait_set(**inputs)

    def _add_output_traits(self, base):
        return add_traits(base, self._fields)

    def _list_outputs(self):
        if self._fields and self._mandatory_inputs:
            for key in self._fields:
                value = getattr(self.inputs, key)
                if not isdefined(value):
                    msg = "%s requires a value for input '%s' because it was listed in 'fields'.                     You can turn off mandatory inputs checking by passing mandatory_inputs = False to the constructor." % (self.__class__.__name__, key)
                    raise ValueError(msg)
        outputs = self._outputs().get()
        for key in self._fields:
            val = getattr(self.inputs, key)
            if isdefined(val):
                outputs[key] = val
        return outputs