import numpy as np
import tempfile
from ...utils.misc import package_check
from ...utils.filemanip import fname_presuffix
from .base import NitimeBaseInterface
from ..base import (
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