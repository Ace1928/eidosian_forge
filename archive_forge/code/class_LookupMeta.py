import os
from os import path as op
import string
import errno
from glob import glob
import nibabel as nb
import imghdr
from .base import (
class LookupMeta(BaseInterface):
    """Lookup meta data values from a Nifti with embedded meta data.

    Example
    -------

    >>> from nipype.interfaces import dcmstack
    >>> lookup = dcmstack.LookupMeta()
    >>> lookup.inputs.in_file = 'functional.nii'
    >>> lookup.inputs.meta_keys = {'RepetitionTime' : 'TR',                                    'EchoTime' : 'TE'}
    >>> result = lookup.run() # doctest: +SKIP
    >>> result.outputs.TR # doctest: +SKIP
    9500.0
    >>> result.outputs.TE # doctest: +SKIP
    95.0
    """
    input_spec = LookupMetaInputSpec
    output_spec = DynamicTraitedSpec

    def _make_name_map(self):
        if isinstance(self.inputs.meta_keys, list):
            self._meta_keys = {}
            for key in self.inputs.meta_keys:
                self._meta_keys[key] = key
        else:
            self._meta_keys = self.inputs.meta_keys

    def _outputs(self):
        self._make_name_map()
        outputs = super(LookupMeta, self)._outputs()
        undefined_traits = {}
        for out_name in list(self._meta_keys.values()):
            outputs.add_trait(out_name, traits.Any)
            undefined_traits[out_name] = Undefined
        outputs.trait_set(trait_change_notify=False, **undefined_traits)
        for out_name in list(self._meta_keys.values()):
            _ = getattr(outputs, out_name)
        return outputs

    def _run_interface(self, runtime):
        self._make_name_map()
        nw = NiftiWrapper.from_filename(self.inputs.in_file)
        self.result = {}
        for meta_key, out_name in list(self._meta_keys.items()):
            self.result[out_name] = nw.meta_ext.get_values(meta_key)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs.update(self.result)
        return outputs