import parlai.core.build_data as build_data
import parlai.utils.logging as logging
import os
from PIL import Image
from zipfile import ZipFile
def _init_resnet_cnn(self):
    """
        Lazily initialize preprocessor model.

        When image_mode is one of the ``resnet`` varieties
        """
    cnn_type, layer_num = self._image_mode_switcher()
    CNN = getattr(self.torchvision.models, cnn_type)
    self.netCNN = self.nn.Sequential(*list(CNN(pretrained=True).children())[:layer_num])
    if self.use_cuda:
        self.netCNN.cuda()