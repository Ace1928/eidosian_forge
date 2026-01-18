import os
import time
from typing import List, Optional, Union, TYPE_CHECKING
import torch
from tensorboard.compat import tf
from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto.event_pb2 import Event, SessionLog
from tensorboard.plugins.projector.projector_config_pb2 import ProjectorConfig
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from ._convert_np import make_np
from ._embedding import get_embedding_info, make_mat, make_sprite, make_tsv, write_pbtxt
from ._onnx_graph import load_onnx_graph
from ._pytorch_graph import graph
from ._utils import figure_to_image
from .summary import (
def add_embedding(self, mat, metadata=None, label_img=None, global_step=None, tag='default', metadata_header=None):
    """Add embedding projector data to summary.

        Args:
            mat (torch.Tensor or numpy.ndarray): A matrix which each row is the feature vector of the data point
            metadata (list): A list of labels, each element will be convert to string
            label_img (torch.Tensor): Images correspond to each data point
            global_step (int): Global step value to record
            tag (str): Name for the embedding
        Shape:
            mat: :math:`(N, D)`, where N is number of data and D is feature dimension

            label_img: :math:`(N, C, H, W)`

        Examples::

            import keyword
            import torch
            meta = []
            while len(meta)<100:
                meta = meta+keyword.kwlist # get some strings
            meta = meta[:100]

            for i, v in enumerate(meta):
                meta[i] = v+str(i)

            label_img = torch.rand(100, 3, 10, 32)
            for i in range(100):
                label_img[i]*=i/100.0

            writer.add_embedding(torch.randn(100, 5), metadata=meta, label_img=label_img)
            writer.add_embedding(torch.randn(100, 5), label_img=label_img)
            writer.add_embedding(torch.randn(100, 5), metadata=meta)
        """
    torch._C._log_api_usage_once('tensorboard.logging.add_embedding')
    mat = make_np(mat)
    if global_step is None:
        global_step = 0
    subdir = f'{str(global_step).zfill(5)}/{self._encode(tag)}'
    save_path = os.path.join(self._get_file_writer().get_logdir(), subdir)
    fs = tf.io.gfile
    if fs.exists(save_path):
        if fs.isdir(save_path):
            print('warning: Embedding dir exists, did you set global_step for add_embedding()?')
        else:
            raise Exception(f'Path: `{save_path}` exists, but is a file. Cannot proceed.')
    else:
        fs.makedirs(save_path)
    if metadata is not None:
        assert mat.shape[0] == len(metadata), '#labels should equal with #data points'
        make_tsv(metadata, save_path, metadata_header=metadata_header)
    if label_img is not None:
        assert mat.shape[0] == label_img.shape[0], '#images should equal with #data points'
        make_sprite(label_img, save_path)
    assert mat.ndim == 2, 'mat should be 2D, where mat.size(0) is the number of data points'
    make_mat(mat, save_path)
    if not hasattr(self, '_projector_config'):
        self._projector_config = ProjectorConfig()
    embedding_info = get_embedding_info(metadata, label_img, subdir, global_step, tag)
    self._projector_config.embeddings.extend([embedding_info])
    from google.protobuf import text_format
    config_pbtxt = text_format.MessageToString(self._projector_config)
    write_pbtxt(self._get_file_writer().get_logdir(), config_pbtxt)