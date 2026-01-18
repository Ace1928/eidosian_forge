import os
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .utils import copy2subjdir
class Label2Label(FSCommand):
    """
    Converts a label in one subject's space to a label
    in another subject's space using either talairach or spherical
    as an intermediate registration space.

    If a source mask is used, then the input label must have been
    created from a surface (ie, the vertex numbers are valid). The
    format can be anything supported by mri_convert or curv or paint.
    Vertices in the source label that do not meet threshold in the
    mask will be removed from the label.

    Examples
    --------
    >>> from nipype.interfaces.freesurfer import Label2Label
    >>> l2l = Label2Label()
    >>> l2l.inputs.hemisphere = 'lh'
    >>> l2l.inputs.subject_id = '10335'
    >>> l2l.inputs.sphere_reg = 'lh.pial'
    >>> l2l.inputs.white = 'lh.pial'
    >>> l2l.inputs.source_subject = 'fsaverage'
    >>> l2l.inputs.source_label = 'lh-pial.stl'
    >>> l2l.inputs.source_white = 'lh.pial'
    >>> l2l.inputs.source_sphere_reg = 'lh.pial'
    >>> l2l.cmdline
    'mri_label2label --hemi lh --trglabel lh-pial_converted.stl --regmethod surface --srclabel lh-pial.stl --srcsubject fsaverage --trgsubject 10335'
    """
    _cmd = 'mri_label2label'
    input_spec = Label2LabelInputSpec
    output_spec = Label2LabelOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = os.path.join(self.inputs.subjects_dir, self.inputs.subject_id, 'label', self.inputs.out_file)
        return outputs

    def run(self, **inputs):
        if self.inputs.copy_inputs:
            self.inputs.subjects_dir = os.getcwd()
            if 'subjects_dir' in inputs:
                inputs['subjects_dir'] = self.inputs.subjects_dir
            hemi = self.inputs.hemisphere
            copy2subjdir(self, self.inputs.sphere_reg, 'surf', '{0}.sphere.reg'.format(hemi))
            copy2subjdir(self, self.inputs.white, 'surf', '{0}.white'.format(hemi))
            copy2subjdir(self, self.inputs.source_sphere_reg, 'surf', '{0}.sphere.reg'.format(hemi), subject_id=self.inputs.source_subject)
            copy2subjdir(self, self.inputs.source_white, 'surf', '{0}.white'.format(hemi), subject_id=self.inputs.source_subject)
        label_dir = os.path.join(self.inputs.subjects_dir, self.inputs.subject_id, 'label')
        if not os.path.isdir(label_dir):
            os.makedirs(label_dir)
        return super(Label2Label, self).run(**inputs)