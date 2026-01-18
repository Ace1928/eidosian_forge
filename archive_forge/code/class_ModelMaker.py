from nipype.interfaces.base import (
import os
class ModelMaker(SEMLikeCommandLine):
    """title: Model Maker

    category: Surface Models

    description: Create 3D surface models from segmented data.<p>Models are imported into Slicer under a model hierarchy node in a MRML scene. The model colors are set by the color table associated with the input volume (these colours will only be visible if you load the model scene file).</p><p><b>Create Multiple:</b></p><p>If you specify a list of Labels, it will over ride any start/end label settings.</p><p>If you click<i>Generate All</i>it will over ride the list of labels and any start/end label settings.</p><p><b>Model Maker Settings:</b></p><p>You can set the number of smoothing iterations, target reduction in number of polygons (decimal percentage). Use 0 and 1 if you wish no smoothing nor decimation.<br>You can set the flags to split normals or generate point normals in this pane as well.<br>You can save a copy of the models after intermediate steps (marching cubes, smoothing, and decimation if not joint smoothing, otherwise just after decimation); these models are not saved in the mrml file, turn off deleting temporary files first in the python window:<br><i>slicer.modules.modelmaker.cliModuleLogic().DeleteTemporaryFilesOff()</i></p>

    version: 4.1

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/ModelMaker

    license: slicer4

    contributor: Nicole Aucoin (SPL, BWH), Ron Kikinis (SPL, BWH), Bill Lorensen (GE)

    acknowledgements: This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.
    """
    input_spec = ModelMakerInputSpec
    output_spec = ModelMakerOutputSpec
    _cmd = 'ModelMaker '
    _outputs_filenames = {'modelSceneFile': 'modelSceneFile.mrml'}