import os
import warnings
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class ProbTrackX(FSLCommand):
    """ Use FSL  probtrackx for tractography on bedpostx results

    Examples
    --------

    >>> from nipype.interfaces import fsl
    >>> pbx = fsl.ProbTrackX(samples_base_name='merged', mask='mask.nii',     seed='MASK_average_thal_right.nii', mode='seedmask',     xfm='trans.mat', n_samples=3, n_steps=10, force_dir=True, opd=True,     os2t=True, target_masks = ['targets_MASK1.nii', 'targets_MASK2.nii'],     thsamples='merged_thsamples.nii', fsamples='merged_fsamples.nii',     phsamples='merged_phsamples.nii', out_dir='.')
    >>> pbx.cmdline
    'probtrackx --forcedir -m mask.nii --mode=seedmask --nsamples=3 --nsteps=10 --opd --os2t --dir=. --samples=merged --seed=MASK_average_thal_right.nii --targetmasks=targets.txt --xfm=trans.mat'

    """
    _cmd = 'probtrackx'
    input_spec = ProbTrackXInputSpec
    output_spec = ProbTrackXOutputSpec

    def __init__(self, **inputs):
        warnings.warn('Deprecated: Please use create_bedpostx_pipeline instead', DeprecationWarning)
        return super(ProbTrackX, self).__init__(**inputs)

    def _run_interface(self, runtime):
        for i in range(1, len(self.inputs.thsamples) + 1):
            _, _, ext = split_filename(self.inputs.thsamples[i - 1])
            copyfile(self.inputs.thsamples[i - 1], self.inputs.samples_base_name + '_th%dsamples' % i + ext, copy=False)
            _, _, ext = split_filename(self.inputs.thsamples[i - 1])
            copyfile(self.inputs.phsamples[i - 1], self.inputs.samples_base_name + '_ph%dsamples' % i + ext, copy=False)
            _, _, ext = split_filename(self.inputs.thsamples[i - 1])
            copyfile(self.inputs.fsamples[i - 1], self.inputs.samples_base_name + '_f%dsamples' % i + ext, copy=False)
        if isdefined(self.inputs.target_masks):
            f = open('targets.txt', 'w')
            for target in self.inputs.target_masks:
                f.write('%s\n' % target)
            f.close()
        if isinstance(self.inputs.seed, list):
            f = open('seeds.txt', 'w')
            for seed in self.inputs.seed:
                if isinstance(seed, list):
                    f.write('%s\n' % ' '.join([str(s) for s in seed]))
                else:
                    f.write('%s\n' % seed)
            f.close()
        runtime = super(ProbTrackX, self)._run_interface(runtime)
        if runtime.stderr:
            self.raise_exception(runtime)
        return runtime

    def _format_arg(self, name, spec, value):
        if name == 'target_masks' and isdefined(value):
            fname = 'targets.txt'
            return super(ProbTrackX, self)._format_arg(name, spec, [fname])
        elif name == 'seed' and isinstance(value, list):
            fname = 'seeds.txt'
            return super(ProbTrackX, self)._format_arg(name, spec, fname)
        else:
            return super(ProbTrackX, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_dir):
            out_dir = self._gen_filename('out_dir')
        else:
            out_dir = self.inputs.out_dir
        outputs['log'] = os.path.abspath(os.path.join(out_dir, 'probtrackx.log'))
        if isdefined(self.inputs.opd is True):
            if isinstance(self.inputs.seed, list) and isinstance(self.inputs.seed[0], list):
                outputs['fdt_paths'] = []
                for seed in self.inputs.seed:
                    outputs['fdt_paths'].append(os.path.abspath(self._gen_fname('fdt_paths_%s' % '_'.join([str(s) for s in seed]), cwd=out_dir, suffix='')))
            else:
                outputs['fdt_paths'] = os.path.abspath(self._gen_fname('fdt_paths', cwd=out_dir, suffix=''))
        if isdefined(self.inputs.target_masks):
            outputs['targets'] = []
            for target in self.inputs.target_masks:
                outputs['targets'].append(os.path.abspath(self._gen_fname('seeds_to_' + os.path.split(target)[1], cwd=out_dir, suffix='')))
        if isdefined(self.inputs.verbose) and self.inputs.verbose == 2:
            outputs['particle_files'] = [os.path.abspath(os.path.join(out_dir, 'particle%d' % i)) for i in range(self.inputs.n_samples)]
        return outputs

    def _gen_filename(self, name):
        if name == 'out_dir':
            return os.getcwd()
        elif name == 'mode':
            if isinstance(self.inputs.seed, list) and isinstance(self.inputs.seed[0], list):
                return 'simple'
            else:
                return 'seedmask'