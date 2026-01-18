import pickle
import os.path as op
import numpy as np
import nibabel as nb
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
class ROIGen(BaseInterface):
    """
    Generates a ROI file for connectivity mapping and a dictionary file containing relevant node information

    Example
    -------

    >>> import nipype.interfaces.cmtk as cmtk
    >>> rg = cmtk.ROIGen()
    >>> rg.inputs.aparc_aseg_file = 'aparc+aseg.nii'
    >>> rg.inputs.use_freesurfer_LUT = True
    >>> rg.inputs.freesurfer_dir = '/usr/local/freesurfer'
    >>> rg.run() # doctest: +SKIP

    The label dictionary is written to disk using Pickle. Resulting data can be loaded using:

    >>> file = open("FreeSurferColorLUT_adapted_aparc+aseg_out.pck", "r")
    >>> file = open("fsLUT_aparc+aseg.pck", "r")
    >>> labelDict = pickle.load(file) # doctest: +SKIP
    >>> labelDict                     # doctest: +SKIP
    """
    input_spec = ROIGenInputSpec
    output_spec = ROIGenOutputSpec

    def _run_interface(self, runtime):
        aparc_aseg_file = self.inputs.aparc_aseg_file
        aparcpath, aparcname, aparcext = split_filename(aparc_aseg_file)
        iflogger.info('Using Aparc+Aseg file: %s', aparcname + aparcext)
        niiAPARCimg = nb.load(aparc_aseg_file)
        niiAPARCdata = np.asanyarray(niiAPARCimg.dataobj)
        niiDataLabels = np.unique(niiAPARCdata)
        numDataLabels = np.size(niiDataLabels)
        iflogger.info('Number of labels in image: %s', numDataLabels)
        write_dict = True
        if self.inputs.use_freesurfer_LUT:
            self.LUT_file = self.inputs.freesurfer_dir + '/FreeSurferColorLUT.txt'
            iflogger.info('Using Freesurfer LUT: %s', self.LUT_file)
            prefix = 'fsLUT'
        elif not self.inputs.use_freesurfer_LUT and isdefined(self.inputs.LUT_file):
            self.LUT_file = op.abspath(self.inputs.LUT_file)
            lutpath, lutname, lutext = split_filename(self.LUT_file)
            iflogger.info('Using Custom LUT file: %s', lutname + lutext)
            prefix = lutname
        else:
            prefix = 'hardcoded'
            write_dict = False
        if isdefined(self.inputs.out_roi_file):
            roi_file = op.abspath(self.inputs.out_roi_file)
        else:
            roi_file = op.abspath(prefix + '_' + aparcname + '.nii')
        if isdefined(self.inputs.out_dict_file):
            dict_file = op.abspath(self.inputs.out_dict_file)
        else:
            dict_file = op.abspath(prefix + '_' + aparcname + '.pck')
        if write_dict:
            iflogger.info('Lookup table: %s', op.abspath(self.LUT_file))
            LUTlabelsRGBA = np.loadtxt(self.LUT_file, skiprows=4, usecols=[0, 1, 2, 3, 4, 5], comments='#', dtype={'names': ('index', 'label', 'R', 'G', 'B', 'A'), 'formats': ('int', '|S30', 'int', 'int', 'int', 'int')})
            numLUTLabels = np.size(LUTlabelsRGBA)
            if numLUTLabels < numDataLabels:
                iflogger.error('LUT file provided does not contain all of the regions in the image')
                iflogger.error('Removing unmapped regions')
            iflogger.info('Number of labels in LUT: %s', numLUTLabels)
            LUTlabelDict = {}
            ' Create dictionary for input LUT table'
            for labels in range(0, numLUTLabels):
                LUTlabelDict[LUTlabelsRGBA[labels][0]] = [LUTlabelsRGBA[labels][1], LUTlabelsRGBA[labels][2], LUTlabelsRGBA[labels][3], LUTlabelsRGBA[labels][4], LUTlabelsRGBA[labels][5]]
            iflogger.info('Printing LUT label dictionary')
            iflogger.info(LUTlabelDict)
        mapDict = {}
        MAPPING = [[1, 2012], [2, 2019], [3, 2032], [4, 2014], [5, 2020], [6, 2018], [7, 2027], [8, 2028], [9, 2003], [10, 2024], [11, 2017], [12, 2026], [13, 2002], [14, 2023], [15, 2010], [16, 2022], [17, 2031], [18, 2029], [19, 2008], [20, 2025], [21, 2005], [22, 2021], [23, 2011], [24, 2013], [25, 2007], [26, 2016], [27, 2006], [28, 2033], [29, 2009], [30, 2015], [31, 2001], [32, 2030], [33, 2034], [34, 2035], [35, 49], [36, 50], [37, 51], [38, 52], [39, 58], [40, 53], [41, 54], [42, 1012], [43, 1019], [44, 1032], [45, 1014], [46, 1020], [47, 1018], [48, 1027], [49, 1028], [50, 1003], [51, 1024], [52, 1017], [53, 1026], [54, 1002], [55, 1023], [56, 1010], [57, 1022], [58, 1031], [59, 1029], [60, 1008], [61, 1025], [62, 1005], [63, 1021], [64, 1011], [65, 1013], [66, 1007], [67, 1016], [68, 1006], [69, 1033], [70, 1009], [71, 1015], [72, 1001], [73, 1030], [74, 1034], [75, 1035], [76, 10], [77, 11], [78, 12], [79, 13], [80, 26], [81, 17], [82, 18], [83, 16]]
        ' Create empty grey matter mask, Populate with only those regions defined in the mapping.'
        niiGM = np.zeros(niiAPARCdata.shape, dtype=np.uint)
        for ma in MAPPING:
            niiGM[niiAPARCdata == ma[1]] = ma[0]
            mapDict[ma[0]] = ma[1]
        iflogger.info('Grey matter mask created')
        greyMaskLabels = np.unique(niiGM)
        numGMLabels = np.size(greyMaskLabels)
        iflogger.info('Number of grey matter labels: %s', numGMLabels)
        labelDict = {}
        GMlabelDict = {}
        for label in greyMaskLabels:
            try:
                mapDict[label]
                if write_dict:
                    GMlabelDict['originalID'] = mapDict[label]
            except:
                iflogger.info('Label %s not in provided mapping', label)
            if write_dict:
                del GMlabelDict
                GMlabelDict = {}
                GMlabelDict['labels'] = LUTlabelDict[label][0]
                GMlabelDict['colors'] = [LUTlabelDict[label][1], LUTlabelDict[label][2], LUTlabelDict[label][3]]
                GMlabelDict['a'] = LUTlabelDict[label][4]
                labelDict[label] = GMlabelDict
        roi_image = nb.Nifti1Image(niiGM, niiAPARCimg.affine, niiAPARCimg.header)
        iflogger.info('Saving ROI File to %s', roi_file)
        nb.save(roi_image, roi_file)
        if write_dict:
            iflogger.info('Saving Dictionary File to %s in Pickle format', dict_file)
            with open(dict_file, 'w') as f:
                pickle.dump(labelDict, f)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        if isdefined(self.inputs.out_roi_file):
            outputs['roi_file'] = op.abspath(self.inputs.out_roi_file)
        else:
            outputs['roi_file'] = op.abspath(self._gen_outfilename('nii'))
        if isdefined(self.inputs.out_dict_file):
            outputs['dict_file'] = op.abspath(self.inputs.out_dict_file)
        else:
            outputs['dict_file'] = op.abspath(self._gen_outfilename('pck'))
        return outputs

    def _gen_outfilename(self, ext):
        _, name, _ = split_filename(self.inputs.aparc_aseg_file)
        if self.inputs.use_freesurfer_LUT:
            prefix = 'fsLUT'
        elif not self.inputs.use_freesurfer_LUT and isdefined(self.inputs.LUT_file):
            lutpath, lutname, lutext = split_filename(self.inputs.LUT_file)
            prefix = lutname
        else:
            prefix = 'hardcoded'
        return prefix + '_' + name + '.' + ext