import json
import logging
import random
import warnings
import numpy as np
from ..base import numeric_types
from .. import ndarray as nd
from ..ndarray._internal import _cvcopyMakeBorder as copyMakeBorder
from .. import io
from .image import RandomOrderAug, ColorJitterAug, LightingAug, ColorNormalizeAug
from .image import ResizeAug, ForceResizeAug, CastAug, HueJitterAug, RandomGrayAug
from .image import fixed_crop, ImageIter, Augmenter
from ..util import is_np_array
from .. import numpy as _mx_np  # pylint: disable=reimported
class ImageDetIter(ImageIter):
    """Image iterator with a large number of augmentation choices for detection.

    Parameters
    ----------
    aug_list : list or None
        Augmenter list for generating distorted images
    batch_size : int
        Number of examples per batch.
    data_shape : tuple
        Data shape in (channels, height, width) format.
        For now, only RGB image with 3 channels is supported.
    path_imgrec : str
        Path to image record file (.rec).
        Created with tools/im2rec.py or bin/im2rec.
    path_imglist : str
        Path to image list (.lst).
        Created with tools/im2rec.py or with custom script.
        Format: Tab separated record of index, one or more labels and relative_path_from_root.
    imglist: list
        A list of images with the label(s).
        Each item is a list [imagelabel: float or list of float, imgpath].
    path_root : str
        Root folder of image files.
    path_imgidx : str
        Path to image index file. Needed for partition and shuffling when using .rec source.
    shuffle : bool
        Whether to shuffle all images at the start of each iteration or not.
        Can be slow for HDD.
    part_index : int
        Partition index.
    num_parts : int
        Total number of partitions.
    data_name : str
        Data name for provided symbols.
    label_name : str
        Name for detection labels
    last_batch_handle : str, optional
        How to handle the last batch.
        This parameter can be 'pad'(default), 'discard' or 'roll_over'.
        If 'pad', the last batch will be padded with data starting from the begining
        If 'discard', the last batch will be discarded
        If 'roll_over', the remaining elements will be rolled over to the next iteration
    kwargs : ...
        More arguments for creating augmenter. See mx.image.CreateDetAugmenter.
    """

    def __init__(self, batch_size, data_shape, path_imgrec=None, path_imglist=None, path_root=None, path_imgidx=None, shuffle=False, part_index=0, num_parts=1, aug_list=None, imglist=None, data_name='data', label_name='label', last_batch_handle='pad', **kwargs):
        super(ImageDetIter, self).__init__(batch_size=batch_size, data_shape=data_shape, path_imgrec=path_imgrec, path_imglist=path_imglist, path_root=path_root, path_imgidx=path_imgidx, shuffle=shuffle, part_index=part_index, num_parts=num_parts, aug_list=[], imglist=imglist, data_name=data_name, label_name=label_name, last_batch_handle=last_batch_handle)
        if aug_list is None:
            self.auglist = CreateDetAugmenter(data_shape, **kwargs)
        else:
            self.auglist = aug_list
        label_shape = self._estimate_label_shape()
        self.provide_label = [(label_name, (self.batch_size, label_shape[0], label_shape[1]))]
        self.label_shape = label_shape

    def _check_valid_label(self, label):
        """Validate label and its shape."""
        if len(label.shape) != 2 or label.shape[1] < 5:
            msg = 'Label with shape (1+, 5+) required, %s received.' % str(label)
            raise RuntimeError(msg)
        valid_label = np.where(np.logical_and(label[:, 0] >= 0, label[:, 3] > label[:, 1], label[:, 4] > label[:, 2]))[0]
        if valid_label.size < 1:
            raise RuntimeError('Invalid label occurs.')

    def _estimate_label_shape(self):
        """Helper function to estimate label shape"""
        max_count = 0
        self.reset()
        try:
            while True:
                label, _ = self.next_sample()
                label = self._parse_label(label)
                max_count = max(max_count, label.shape[0])
        except StopIteration:
            pass
        self.reset()
        return (max_count, label.shape[1])

    def _parse_label(self, label):
        """Helper function to parse object detection label.

        Format for raw label:
        n 	 k 	 ... 	 [id 	 xmin	 ymin 	 xmax 	 ymax 	 ...] 	 [repeat]
        where n is the width of header, 2 or larger
        k is the width of each object annotation, can be arbitrary, at least 5
        """
        if isinstance(label, nd.NDArray):
            label = label.asnumpy()
        raw = label.ravel()
        if raw.size < 7:
            raise RuntimeError('Label shape is invalid: ' + str(raw.shape))
        header_width = int(raw[0])
        obj_width = int(raw[1])
        if (raw.size - header_width) % obj_width != 0:
            msg = 'Label shape %s inconsistent with annotation width %d.' % (str(raw.shape), obj_width)
            raise RuntimeError(msg)
        out = np.reshape(raw[header_width:], (-1, obj_width))
        valid = np.where(np.logical_and(out[:, 3] > out[:, 1], out[:, 4] > out[:, 2]))[0]
        if valid.size < 1:
            raise RuntimeError('Encounter sample with no valid label.')
        return out[valid, :]

    def reshape(self, data_shape=None, label_shape=None):
        """Reshape iterator for data_shape or label_shape.

        Parameters
        ----------
        data_shape : tuple or None
            Reshape the data_shape to the new shape if not None
        label_shape : tuple or None
            Reshape label shape to new shape if not None
        """
        if data_shape is not None:
            self.check_data_shape(data_shape)
            self.provide_data = [(self.provide_data[0][0], (self.batch_size,) + data_shape)]
            self.data_shape = data_shape
        if label_shape is not None:
            self.check_label_shape(label_shape)
            self.provide_label = [(self.provide_label[0][0], (self.batch_size,) + label_shape)]
            self.label_shape = label_shape

    def _batchify(self, batch_data, batch_label, start=0):
        """Override the helper function for batchifying data"""
        i = start
        batch_size = self.batch_size
        array_fn = _mx_np.array if is_np_array() else nd.array
        try:
            while i < batch_size:
                label, s = self.next_sample()
                data = self.imdecode(s)
                try:
                    self.check_valid_image([data])
                    label = self._parse_label(label)
                    data, label = self.augmentation_transform(data, label)
                    self._check_valid_label(label)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue
                for datum in [data]:
                    assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                    batch_data[i] = self.postprocess_data(datum)
                    num_object = label.shape[0]
                    batch_label[i][0:num_object] = array_fn(label)
                    if num_object < batch_label[i].shape[0]:
                        batch_label[i][num_object:] = -1
                    i += 1
        except StopIteration:
            if not i:
                raise StopIteration
        return i

    def next(self):
        """Override the function for returning next batch."""
        batch_size = self.batch_size
        c, h, w = self.data_shape
        if self._cache_data is not None:
            assert self._cache_label is not None, "_cache_label didn't have values"
            assert self._cache_idx is not None, "_cache_idx didn't have values"
            batch_data = self._cache_data
            batch_label = self._cache_label
            i = self._cache_idx
        else:
            if is_np_array():
                zeros_fn = _mx_np.zeros
                empty_fn = _mx_np.empty
            else:
                zeros_fn = nd.zeros
                empty_fn = nd.empty
            batch_data = zeros_fn((batch_size, c, h, w))
            batch_label = empty_fn(self.provide_label[0][1])
            batch_label[:] = -1
            i = self._batchify(batch_data, batch_label)
        pad = batch_size - i
        if pad != 0:
            if self.last_batch_handle == 'discard':
                raise StopIteration
            if self.last_batch_handle == 'roll_over' and self._cache_data is None:
                self._cache_data = batch_data
                self._cache_label = batch_label
                self._cache_idx = i
                raise StopIteration
            _ = self._batchify(batch_data, batch_label, i)
            if self.last_batch_handle == 'pad':
                self._allow_read = False
            else:
                self._cache_data = None
                self._cache_label = None
                self._cache_idx = None
        return io.DataBatch([batch_data], [batch_label], pad=pad)

    def augmentation_transform(self, data, label):
        """Override Transforms input data with specified augmentations."""
        for aug in self.auglist:
            data, label = aug(data, label)
        return (data, label)

    def check_label_shape(self, label_shape):
        """Checks if the new label shape is valid"""
        if not len(label_shape) == 2:
            raise ValueError('label_shape should have length 2')
        if label_shape[0] < self.label_shape[0]:
            msg = 'Attempts to reduce label count from %d to %d, not allowed.' % (self.label_shape[0], label_shape[0])
            raise ValueError(msg)
        if label_shape[1] != self.provide_label[0][1][2]:
            msg = 'label_shape object width inconsistent: %d vs %d.' % (self.provide_label[0][1][2], label_shape[1])
            raise ValueError(msg)

    def draw_next(self, color=None, thickness=2, mean=None, std=None, clip=True, waitKey=None, window_name='draw_next', id2labels=None):
        """Display next image with bounding boxes drawn.

        Parameters
        ----------
        color : tuple
            Bounding box color in RGB, use None for random color
        thickness : int
            Bounding box border thickness
        mean : True or numpy.ndarray
            Compensate for the mean to have better visual effect
        std : True or numpy.ndarray
            Revert standard deviations
        clip : bool
            If true, clip to [0, 255] for better visual effect
        waitKey : None or int
            Hold the window for waitKey milliseconds if set, skip ploting if None
        window_name : str
            Plot window name if waitKey is set.
        id2labels : dict
            Mapping of labels id to labels name.

        Returns
        -------
            numpy.ndarray

        Examples
        --------
        >>> # use draw_next to get images with bounding boxes drawn
        >>> iterator = mx.image.ImageDetIter(1, (3, 600, 600), path_imgrec='train.rec')
        >>> for image in iterator.draw_next(waitKey=None):
        ...     # display image
        >>> # or let draw_next display using cv2 module
        >>> for image in iterator.draw_next(waitKey=0, window_name='disp'):
        ...     pass
        """
        try:
            import cv2
        except ImportError as e:
            warnings.warn('Unable to import cv2, skip drawing: %s', str(e))
            return
        count = 0
        try:
            while True:
                label, s = self.next_sample()
                data = self.imdecode(s)
                try:
                    self.check_valid_image([data])
                    label = self._parse_label(label)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue
                count += 1
                data, label = self.augmentation_transform(data, label)
                image = data.asnumpy()
                if std is True:
                    std = np.array([58.395, 57.12, 57.375])
                elif std is not None:
                    assert isinstance(std, np.ndarray) and std.shape[0] in [1, 3]
                if std is not None:
                    image *= std
                if mean is True:
                    mean = np.array([123.68, 116.28, 103.53])
                elif mean is not None:
                    assert isinstance(mean, np.ndarray) and mean.shape[0] in [1, 3]
                if mean is not None:
                    image += mean
                image[:, :, (0, 1, 2)] = image[:, :, (2, 1, 0)]
                if clip:
                    image = np.maximum(0, np.minimum(255, image))
                if color:
                    color = color[::-1]
                image = image.astype(np.uint8)
                height, width, _ = image.shape
                for i in range(label.shape[0]):
                    x1 = int(label[i, 1] * width)
                    if x1 < 0:
                        continue
                    y1 = int(label[i, 2] * height)
                    x2 = int(label[i, 3] * width)
                    y2 = int(label[i, 4] * height)
                    bc = np.random.rand(3) * 255 if not color else color
                    cv2.rectangle(image, (x1, y1), (x2, y2), bc, thickness)
                    if id2labels is not None:
                        cls_id = int(label[i, 0])
                        if cls_id in id2labels:
                            cls_name = id2labels[cls_id]
                            text = '{:s}'.format(cls_name)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.5
                            text_height = cv2.getTextSize(text, font, font_scale, 2)[0][1]
                            tc = (255, 255, 255)
                            tpos = (x1 + 5, y1 + text_height + 5)
                            cv2.putText(image, text, tpos, font, font_scale, tc, 2)
                if waitKey is not None:
                    cv2.imshow(window_name, image)
                    cv2.waitKey(waitKey)
                yield image
        except StopIteration:
            if not count:
                return

    def sync_label_shape(self, it, verbose=False):
        """Synchronize label shape with the input iterator. This is useful when
        train/validation iterators have different label padding.

        Parameters
        ----------
        it : ImageDetIter
            The other iterator to synchronize
        verbose : bool
            Print verbose log if true

        Returns
        -------
        ImageDetIter
            The synchronized other iterator, the internal label shape is updated as well.

        Examples
        --------
        >>> train_iter = mx.image.ImageDetIter(32, (3, 300, 300), path_imgrec='train.rec')
        >>> val_iter = mx.image.ImageDetIter(32, (3, 300, 300), path.imgrec='val.rec')
        >>> train_iter.label_shape
        (30, 6)
        >>> val_iter.label_shape
        (25, 6)
        >>> val_iter = train_iter.sync_label_shape(val_iter, verbose=False)
        >>> train_iter.label_shape
        (30, 6)
        >>> val_iter.label_shape
        (30, 6)
        """
        assert isinstance(it, ImageDetIter), 'Synchronize with invalid iterator.'
        train_label_shape = self.label_shape
        val_label_shape = it.label_shape
        assert train_label_shape[1] == val_label_shape[1], 'object width mismatch.'
        max_count = max(train_label_shape[0], val_label_shape[0])
        if max_count > train_label_shape[0]:
            self.reshape(None, (max_count, train_label_shape[1]))
        if max_count > val_label_shape[0]:
            it.reshape(None, (max_count, val_label_shape[1]))
        if verbose and max_count > min(train_label_shape[0], val_label_shape[0]):
            logging.info('Resized label_shape to (%d, %d).', max_count, train_label_shape[1])
        return it