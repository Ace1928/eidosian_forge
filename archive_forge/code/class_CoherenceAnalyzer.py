import numpy as np
import tempfile
from ...utils.misc import package_check
from ...utils.filemanip import fname_presuffix
from .base import NitimeBaseInterface
from ..base import (
class CoherenceAnalyzer(NitimeBaseInterface):
    """Wraps nitime.analysis.CoherenceAnalyzer: Coherence/y"""
    input_spec = CoherenceAnalyzerInputSpec
    output_spec = CoherenceAnalyzerOutputSpec

    def _read_csv(self):
        """
        Read from csv in_file and return an array and ROI names

        The input file should have a first row containing the names of the
        ROIs (strings)

        the rest of the data will be read in and transposed so that the rows
        (TRs) will becomes the second (and last) dimension of the array

        """
        first_row = open(self.inputs.in_file).readline()
        if not first_row[1].isalpha():
            raise ValueError('First row of in_file should contain ROI names as strings of characters')
        roi_names = open(self.inputs.in_file).readline().replace('"', '').strip('\n').split(',')
        data = np.loadtxt(self.inputs.in_file, skiprows=1, delimiter=',').T
        return (data, roi_names)

    def _csv2ts(self):
        """Read data from the in_file and generate a nitime TimeSeries object"""
        from nitime.timeseries import TimeSeries
        data, roi_names = self._read_csv()
        TS = TimeSeries(data=data, sampling_interval=self.inputs.TR, time_unit='s')
        TS.metadata = dict(ROIs=roi_names)
        return TS

    def _run_interface(self, runtime):
        import nitime.analysis as nta
        lb, ub = self.inputs.frequency_range
        if self.inputs.in_TS is Undefined:
            TS = self._csv2ts()
        else:
            TS = self.inputs.in_TS
        if 'ROIs' not in TS.metadata:
            self.ROIs = ['roi_%d' % x for x, _ in enumerate(TS.data)]
        else:
            self.ROIs = TS.metadata['ROIs']
        A = nta.CoherenceAnalyzer(TS, method=dict(this_method='welch', NFFT=self.inputs.NFFT, n_overlap=self.inputs.n_overlap))
        freq_idx = np.where((A.frequencies > self.inputs.frequency_range[0]) * (A.frequencies < self.inputs.frequency_range[1]))[0]
        self.coherence = np.mean(A.coherence[:, :, freq_idx], -1)
        self.delay = np.mean(A.delay[:, :, freq_idx], -1)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['coherence_array'] = self.coherence
        outputs['timedelay_array'] = self.delay
        if isdefined(self.inputs.output_csv_file) and hasattr(self, 'coherence'):
            self._make_output_files()
            outputs['coherence_csv'] = fname_presuffix(self.inputs.output_csv_file, suffix='_coherence')
            outputs['timedelay_csv'] = fname_presuffix(self.inputs.output_csv_file, suffix='_delay')
        if isdefined(self.inputs.output_figure_file) and hasattr(self, 'coherence'):
            self._make_output_figures()
            outputs['coherence_fig'] = fname_presuffix(self.inputs.output_figure_file, suffix='_coherence')
            outputs['timedelay_fig'] = fname_presuffix(self.inputs.output_figure_file, suffix='_delay')
        return outputs

    def _make_output_files(self):
        """
        Generate the output csv files.
        """
        for this in zip([self.coherence, self.delay], ['coherence', 'delay']):
            tmp_f = tempfile.mkstemp()[1]
            np.savetxt(tmp_f, this[0], delimiter=',')
            fid = open(fname_presuffix(self.inputs.output_csv_file, suffix='_%s' % this[1]), 'w+')
            fid.write(',' + ','.join(self.ROIs) + '\n')
            for r, line in zip(self.ROIs, open(tmp_f)):
                fid.write('%s,%s' % (r, line))
            fid.close()

    def _make_output_figures(self):
        """
        Generate the desired figure and save the files according to
        self.inputs.output_figure_file

        """
        import nitime.viz as viz
        if self.inputs.figure_type == 'matrix':
            fig_coh = viz.drawmatrix_channels(self.coherence, channel_names=self.ROIs, color_anchor=0)
            fig_coh.savefig(fname_presuffix(self.inputs.output_figure_file, suffix='_coherence'))
            fig_dt = viz.drawmatrix_channels(self.delay, channel_names=self.ROIs, color_anchor=0)
            fig_dt.savefig(fname_presuffix(self.inputs.output_figure_file, suffix='_delay'))
        else:
            fig_coh = viz.drawgraph_channels(self.coherence, channel_names=self.ROIs)
            fig_coh.savefig(fname_presuffix(self.inputs.output_figure_file, suffix='_coherence'))
            fig_dt = viz.drawgraph_channels(self.delay, channel_names=self.ROIs)
            fig_dt.savefig(fname_presuffix(self.inputs.output_figure_file, suffix='_delay'))