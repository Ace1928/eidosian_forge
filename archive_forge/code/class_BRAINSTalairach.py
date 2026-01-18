import os
from ...base import (
class BRAINSTalairach(SEMLikeCommandLine):
    """title: BRAINS Talairach

    category: BRAINS.Segmentation

    description: This program creates a VTK structured grid defining the Talairach coordinate system based on four points: AC, PC, IRP, and SLA. The resulting structured grid can be written as either a classic VTK file or the new VTK XML file format. Two representations of the resulting grid can be written. The first is a bounding box representation that also contains the location of the AC and PC points. The second representation is the full Talairach grid representation that includes the additional rows of boxes added to the inferior allowing full coverage of the cerebellum.

    version: 0.1

    documentation-url: http://www.nitrc.org/plugins/mwiki/index.php/brains:BRAINSTalairach

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: Steven Dunn and Vincent Magnotta

    acknowledgements: Funding for this work was provided by NIH/NINDS award NS050568
    """
    input_spec = BRAINSTalairachInputSpec
    output_spec = BRAINSTalairachOutputSpec
    _cmd = ' BRAINSTalairach '
    _outputs_filenames = {'outputGrid': 'outputGrid', 'outputBox': 'outputBox'}
    _redirect_x = False