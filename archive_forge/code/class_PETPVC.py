import os
from .base import (
from ..utils.filemanip import fname_presuffix
from ..external.due import BibTeX
class PETPVC(CommandLine):
    """Use PETPVC for partial volume correction of PET images.

    PETPVC ([1]_, [2]_) is a software from the Nuclear Medicine Department
    of the UCL University Hospital, London, UK.

    Examples
    --------
    >>> from ..testing import example_data
    >>> #TODO get data for PETPVC
    >>> pvc = PETPVC()
    >>> pvc.inputs.in_file   = 'pet.nii.gz'
    >>> pvc.inputs.mask_file = 'tissues.nii.gz'
    >>> pvc.inputs.out_file  = 'pet_pvc_rbv.nii.gz'
    >>> pvc.inputs.pvc = 'RBV'
    >>> pvc.inputs.fwhm_x = 2.0
    >>> pvc.inputs.fwhm_y = 2.0
    >>> pvc.inputs.fwhm_z = 2.0
    >>> outs = pvc.run() #doctest: +SKIP

    References
    ----------
    .. [1] K. Erlandsson, I. Buvat, P. H. Pretorius, B. A. Thomas, and B. F. Hutton,
           "A review of partial volume correction techniques for emission tomography
           and their applications in neurology, cardiology and oncology," Phys. Med.
           Biol., vol. 57, no. 21, p. R119, 2012.
    .. [2] https://github.com/UCL/PETPVC

    """
    input_spec = PETPVCInputSpec
    output_spec = PETPVCOutputSpec
    _cmd = 'petpvc'
    _references = [{'entry': BibTeX('@article{0031-9155-61-22-7975,author={Benjamin A Thomas and Vesna Cuplov and Alexandre Bousse and Adriana Mendes and Kris Thielemans and Brian F Hutton and Kjell Erlandsson},title={PETPVC: a toolbox for performing partial volume correction techniques in positron emission tomography},journal={Physics in Medicine and Biology},volume={61},number={22},pages={7975},url={http://stacks.iop.org/0031-9155/61/i=22/a=7975},doi={https://doi.org/10.1088/0031-9155/61/22/7975},year={2016},}'), 'description': 'PETPVC software implementation publication', 'tags': ['implementation']}]

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self.inputs.out_file
        if not isdefined(outputs['out_file']):
            method_name = self.inputs.pvc.lower()
            outputs['out_file'] = self._gen_fname(self.inputs.in_file, suffix='_{}_pvc'.format(method_name))
        outputs['out_file'] = os.path.abspath(outputs['out_file'])
        return outputs

    def _gen_fname(self, basename, cwd=None, suffix=None, change_ext=True, ext='.nii.gz'):
        """Generate a filename based on the given parameters.

        The filename will take the form: cwd/basename<suffix><ext>.
        If change_ext is True, it will use the extensions specified in
        <instance>inputs.output_type.

        Parameters
        ----------
        basename : str
            Filename to base the new filename on.
        cwd : str
            Path to prefix to the new filename. (default is os.getcwd())
        suffix : str
            Suffix to add to the `basename`.  (defaults is '' )
        change_ext : bool
            Flag to change the filename extension to the given `ext`.
            (Default is False)

        Returns
        -------
        fname : str
            New filename based on given parameters.

        """
        if basename == '':
            msg = 'Unable to generate filename for command %s. ' % self.cmd
            msg += 'basename is not set!'
            raise ValueError(msg)
        if cwd is None:
            cwd = os.getcwd()
        if change_ext:
            if suffix:
                suffix = ''.join((suffix, ext))
            else:
                suffix = ext
        if suffix is None:
            suffix = ''
        fname = fname_presuffix(basename, suffix=suffix, use_ext=False, newpath=cwd)
        return fname

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()['out_file']
        return None