from ..base import (
import os
class AccuracyTester(CommandLine):
    """
    Test the accuracy of an existing training dataset on a set of hand-labelled subjects.
    Note: This may or may not be working. Couldn't presently not confirm because fix fails on this (even outside of nipype) without leaving an error msg.
    """
    input_spec = AccuracyTesterInputSpec
    output_spec = AccuracyTesterOutputSpec
    cmd = 'fix -C'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if isdefined(self.inputs.output_directory):
            outputs['output_directory'] = Directory(exists=False, value=self.inputs.output_directory)
        else:
            outputs['output_directory'] = Directory(exists=False, value='accuracy_test')
        return outputs