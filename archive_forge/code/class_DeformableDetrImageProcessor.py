import io
import pathlib
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils import BaseImageProcessor, get_size_dict
from ...image_transforms import (
from ...image_utils import (
from ...utils import (
class DeformableDetrImageProcessor(BaseImageProcessor):
    """
    Constructs a Deformable DETR image processor.

    Args:
        format (`str`, *optional*, defaults to `"coco_detection"`):
            Data format of the annotations. One of "coco_detection" or "coco_panoptic".
        do_resize (`bool`, *optional*, defaults to `True`):
            Controls whether to resize the image's (height, width) dimensions to the specified `size`. Can be
            overridden by the `do_resize` parameter in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 800, "longest_edge": 1333}`):
            Size of the image's (height, width) dimensions after resizing. Can be overridden by the `size` parameter in
            the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            Resampling filter to use if resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Controls whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_normalize:
            Controls whether to normalize the image. Can be overridden by the `do_normalize` parameter in the
            `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`):
            Mean values to use when normalizing the image. Can be a single value or a list of values, one for each
            channel. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_STD`):
            Standard deviation values to use when normalizing the image. Can be a single value or a list of values, one
            for each channel. Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_annotations (`bool`, *optional*, defaults to `True`):
            Controls whether to convert the annotations to the format expected by the DETR model. Converts the
            bounding boxes to the format `(center_x, center_y, width, height)` and in the range `[0, 1]`.
            Can be overridden by the `do_convert_annotations` parameter in the `preprocess` method.
        do_pad (`bool`, *optional*, defaults to `True`):
            Controls whether to pad the image. Can be overridden by the `do_pad` parameter in the `preprocess`
            method. If `True` will pad the images in the batch to the largest height and width in the batch.
            Padding will be applied to the bottom and right of the image with zeros.
    """
    model_input_names = ['pixel_values', 'pixel_mask']

    def __init__(self, format: Union[str, AnnotationFormat]=AnnotationFormat.COCO_DETECTION, do_resize: bool=True, size: Dict[str, int]=None, resample: PILImageResampling=PILImageResampling.BILINEAR, do_rescale: bool=True, rescale_factor: Union[int, float]=1 / 255, do_normalize: bool=True, image_mean: Union[float, List[float]]=None, image_std: Union[float, List[float]]=None, do_convert_annotations: Optional[bool]=None, do_pad: bool=True, **kwargs) -> None:
        if 'pad_and_return_pixel_mask' in kwargs:
            do_pad = kwargs.pop('pad_and_return_pixel_mask')
        if 'max_size' in kwargs:
            logger.warning_once("The `max_size` parameter is deprecated and will be removed in v4.26. Please specify in `size['longest_edge'] instead`.")
            max_size = kwargs.pop('max_size')
        else:
            max_size = None if size is None else 1333
        size = size if size is not None else {'shortest_edge': 800, 'longest_edge': 1333}
        size = get_size_dict(size, max_size=max_size, default_to_square=False)
        if do_convert_annotations is None:
            do_convert_annotations = do_normalize
        super().__init__(**kwargs)
        self.format = format
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.do_convert_annotations = do_convert_annotations
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.do_pad = do_pad

    @classmethod
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        """
        Overrides the `from_dict` method from the base class to make sure parameters are updated if image processor is
        created using from_dict and kwargs e.g. `DeformableDetrImageProcessor.from_pretrained(checkpoint, size=600,
        max_size=800)`
        """
        image_processor_dict = image_processor_dict.copy()
        if 'max_size' in kwargs:
            image_processor_dict['max_size'] = kwargs.pop('max_size')
        if 'pad_and_return_pixel_mask' in kwargs:
            image_processor_dict['pad_and_return_pixel_mask'] = kwargs.pop('pad_and_return_pixel_mask')
        return super().from_dict(image_processor_dict, **kwargs)

    def prepare_annotation(self, image: np.ndarray, target: Dict, format: Optional[AnnotationFormat]=None, return_segmentation_masks: bool=None, masks_path: Optional[Union[str, pathlib.Path]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None) -> Dict:
        """
        Prepare an annotation for feeding into DeformableDetr model.
        """
        format = format if format is not None else self.format
        if format == AnnotationFormat.COCO_DETECTION:
            return_segmentation_masks = False if return_segmentation_masks is None else return_segmentation_masks
            target = prepare_coco_detection_annotation(image, target, return_segmentation_masks, input_data_format=input_data_format)
        elif format == AnnotationFormat.COCO_PANOPTIC:
            return_segmentation_masks = True if return_segmentation_masks is None else return_segmentation_masks
            target = prepare_coco_panoptic_annotation(image, target, masks_path=masks_path, return_masks=return_segmentation_masks, input_data_format=input_data_format)
        else:
            raise ValueError(f'Format {format} is not supported.')
        return target

    def prepare(self, image, target, return_segmentation_masks=None, masks_path=None):
        logger.warning_once('The `prepare` method is deprecated and will be removed in a v4.33. Please use `prepare_annotation` instead. Note: the `prepare_annotation` method does not return the image anymore.')
        target = self.prepare_annotation(image, target, return_segmentation_masks, masks_path, self.format)
        return (image, target)

    def convert_coco_poly_to_mask(self, *args, **kwargs):
        logger.warning_once('The `convert_coco_poly_to_mask` method is deprecated and will be removed in v4.33. ')
        return convert_coco_poly_to_mask(*args, **kwargs)

    def prepare_coco_detection(self, *args, **kwargs):
        logger.warning_once('The `prepare_coco_detection` method is deprecated and will be removed in v4.33. ')
        return prepare_coco_detection_annotation(*args, **kwargs)

    def prepare_coco_panoptic(self, *args, **kwargs):
        logger.warning_once('The `prepare_coco_panoptic` method is deprecated and will be removed in v4.33. ')
        return prepare_coco_panoptic_annotation(*args, **kwargs)

    def resize(self, image: np.ndarray, size: Dict[str, int], resample: PILImageResampling=PILImageResampling.BILINEAR, data_format: Optional[ChannelDimension]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> np.ndarray:
        """
        Resize the image to the given size. Size can be `min_size` (scalar) or `(height, width)` tuple. If size is an
        int, smaller edge of the image will be matched to this number.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary containing the size to resize to. Can contain the keys `shortest_edge` and `longest_edge` or
                `height` and `width`.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Resampling filter to use if resizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        if 'max_size' in kwargs:
            logger.warning_once("The `max_size` parameter is deprecated and will be removed in v4.26. Please specify in `size['longest_edge'] instead`.")
            max_size = kwargs.pop('max_size')
        else:
            max_size = None
        size = get_size_dict(size, max_size=max_size, default_to_square=False)
        if 'shortest_edge' in size and 'longest_edge' in size:
            size = get_resize_output_image_size(image, size['shortest_edge'], size['longest_edge'], input_data_format=input_data_format)
        elif 'height' in size and 'width' in size:
            size = (size['height'], size['width'])
        else:
            raise ValueError(f"Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys. Got {size.keys()}.")
        image = resize(image, size=size, resample=resample, data_format=data_format, input_data_format=input_data_format, **kwargs)
        return image

    def resize_annotation(self, annotation, orig_size, size, resample: PILImageResampling=PILImageResampling.NEAREST) -> Dict:
        """
        Resize the annotation to match the resized image. If size is an int, smaller edge of the mask will be matched
        to this number.
        """
        return resize_annotation(annotation, orig_size=orig_size, target_size=size, resample=resample)

    def rescale(self, image: np.ndarray, rescale_factor: float, data_format: Optional[Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None) -> np.ndarray:
        """
        Rescale the image by the given factor. image = image * rescale_factor.

        Args:
            image (`np.ndarray`):
                Image to rescale.
            rescale_factor (`float`):
                The value to use for rescaling.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the input image. If unset, is inferred from the input image. Can be
                one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
        """
        return rescale(image, rescale_factor, data_format=data_format, input_data_format=input_data_format)

    def normalize_annotation(self, annotation: Dict, image_size: Tuple[int, int]) -> Dict:
        """
        Normalize the boxes in the annotation from `[top_left_x, top_left_y, bottom_right_x, bottom_right_y]` to
        `[center_x, center_y, width, height]` format and from absolute to relative pixel values.
        """
        return normalize_annotation(annotation, image_size=image_size)

    def _update_annotation_for_padded_image(self, annotation: Dict, input_image_size: Tuple[int, int], output_image_size: Tuple[int, int], padding, update_bboxes) -> Dict:
        """
        Update the annotation for a padded image.
        """
        new_annotation = {}
        new_annotation['size'] = output_image_size
        for key, value in annotation.items():
            if key == 'masks':
                masks = value
                masks = pad(masks, padding, mode=PaddingMode.CONSTANT, constant_values=0, input_data_format=ChannelDimension.FIRST)
                masks = safe_squeeze(masks, 1)
                new_annotation['masks'] = masks
            elif key == 'boxes' and update_bboxes:
                boxes = value
                boxes *= np.asarray([input_image_size[1] / output_image_size[1], input_image_size[0] / output_image_size[0], input_image_size[1] / output_image_size[1], input_image_size[0] / output_image_size[0]])
                new_annotation['boxes'] = boxes
            elif key == 'size':
                new_annotation['size'] = output_image_size
            else:
                new_annotation[key] = value
        return new_annotation

    def _pad_image(self, image: np.ndarray, output_size: Tuple[int, int], annotation: Optional[Dict[str, Any]]=None, constant_values: Union[float, Iterable[float]]=0, data_format: Optional[ChannelDimension]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None, update_bboxes: bool=True) -> np.ndarray:
        """
        Pad an image with zeros to the given size.
        """
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)
        output_height, output_width = output_size
        pad_bottom = output_height - input_height
        pad_right = output_width - input_width
        padding = ((0, pad_bottom), (0, pad_right))
        padded_image = pad(image, padding, mode=PaddingMode.CONSTANT, constant_values=constant_values, data_format=data_format, input_data_format=input_data_format)
        if annotation is not None:
            annotation = self._update_annotation_for_padded_image(annotation, (input_height, input_width), (output_height, output_width), padding, update_bboxes)
        return (padded_image, annotation)

    def pad(self, images: List[np.ndarray], annotations: Optional[Union[AnnotationType, List[AnnotationType]]]=None, constant_values: Union[float, Iterable[float]]=0, return_pixel_mask: bool=True, return_tensors: Optional[Union[str, TensorType]]=None, data_format: Optional[ChannelDimension]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None, update_bboxes: bool=True) -> BatchFeature:
        """
        Pads a batch of images to the bottom and right of the image with zeros to the size of largest height and width
        in the batch and optionally returns their corresponding pixel mask.

        Args:
            images (List[`np.ndarray`]):
                Images to pad.
            annotations (`AnnotationType` or `List[AnnotationType]`, *optional*):
                Annotations to transform according to the padding that is applied to the images.
            constant_values (`float` or `Iterable[float]`, *optional*):
                The value to use for the padding if `mode` is `"constant"`.
            return_pixel_mask (`bool`, *optional*, defaults to `True`):
                Whether to return a pixel mask.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
            update_bboxes (`bool`, *optional*, defaults to `True`):
                Whether to update the bounding boxes in the annotations to match the padded images. If the
                bounding boxes have not been converted to relative coordinates and `(centre_x, centre_y, width, height)`
                format, the bounding boxes will not be updated.
        """
        pad_size = get_max_height_width(images, input_data_format=input_data_format)
        annotation_list = annotations if annotations is not None else [None] * len(images)
        padded_images = []
        padded_annotations = []
        for image, annotation in zip(images, annotation_list):
            padded_image, padded_annotation = self._pad_image(image, pad_size, annotation, constant_values=constant_values, data_format=data_format, input_data_format=input_data_format, update_bboxes=update_bboxes)
            padded_images.append(padded_image)
            padded_annotations.append(padded_annotation)
        data = {'pixel_values': padded_images}
        if return_pixel_mask:
            masks = [make_pixel_mask(image=image, output_size=pad_size, input_data_format=input_data_format) for image in images]
            data['pixel_mask'] = masks
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)
        if annotations is not None:
            encoded_inputs['labels'] = [BatchFeature(annotation, tensor_type=return_tensors) for annotation in padded_annotations]
        return encoded_inputs

    def preprocess(self, images: ImageInput, annotations: Optional[Union[AnnotationType, List[AnnotationType]]]=None, return_segmentation_masks: bool=None, masks_path: Optional[Union[str, pathlib.Path]]=None, do_resize: Optional[bool]=None, size: Optional[Dict[str, int]]=None, resample=None, do_rescale: Optional[bool]=None, rescale_factor: Optional[Union[int, float]]=None, do_normalize: Optional[bool]=None, do_convert_annotations: Optional[bool]=None, image_mean: Optional[Union[float, List[float]]]=None, image_std: Optional[Union[float, List[float]]]=None, do_pad: Optional[bool]=None, format: Optional[Union[str, AnnotationFormat]]=None, return_tensors: Optional[Union[TensorType, str]]=None, data_format: Union[str, ChannelDimension]=ChannelDimension.FIRST, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> BatchFeature:
        """
        Preprocess an image or a batch of images so that it can be used by the model.

        Args:
            images (`ImageInput`):
                Image or batch of images to preprocess. Expects a single or batch of images with pixel values ranging
                from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            annotations (`AnnotationType` or `List[AnnotationType]`, *optional*):
                List of annotations associated with the image or batch of images. If annotation is for object
                detection, the annotations should be a dictionary with the following keys:
                - "image_id" (`int`): The image id.
                - "annotations" (`List[Dict]`): List of annotations for an image. Each annotation should be a
                  dictionary. An image can have no annotations, in which case the list should be empty.
                If annotation is for segmentation, the annotations should be a dictionary with the following keys:
                - "image_id" (`int`): The image id.
                - "segments_info" (`List[Dict]`): List of segments for an image. Each segment should be a dictionary.
                  An image can have no segments, in which case the list should be empty.
                - "file_name" (`str`): The file name of the image.
            return_segmentation_masks (`bool`, *optional*, defaults to self.return_segmentation_masks):
                Whether to return segmentation masks.
            masks_path (`str` or `pathlib.Path`, *optional*):
                Path to the directory containing the segmentation masks.
            do_resize (`bool`, *optional*, defaults to self.do_resize):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to self.size):
                Size of the image after resizing.
            resample (`PILImageResampling`, *optional*, defaults to self.resample):
                Resampling filter to use when resizing the image.
            do_rescale (`bool`, *optional*, defaults to self.do_rescale):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to self.rescale_factor):
                Rescale factor to use when rescaling the image.
            do_normalize (`bool`, *optional*, defaults to self.do_normalize):
                Whether to normalize the image.
            do_convert_annotations (`bool`, *optional*, defaults to self.do_convert_annotations):
                Whether to convert the annotations to the format expected by the model. Converts the bounding
                boxes from the format `(top_left_x, top_left_y, width, height)` to `(center_x, center_y, width, height)`
                and in relative coordinates.
            image_mean (`float` or `List[float]`, *optional*, defaults to self.image_mean):
                Mean to use when normalizing the image.
            image_std (`float` or `List[float]`, *optional*, defaults to self.image_std):
                Standard deviation to use when normalizing the image.
            do_pad (`bool`, *optional*, defaults to self.do_pad):
                Whether to pad the image. If `True` will pad the images in the batch to the largest image in the batch
                and create a pixel mask. Padding will be applied to the bottom and right of the image with zeros.
            format (`str` or `AnnotationFormat`, *optional*, defaults to self.format):
                Format of the annotations.
            return_tensors (`str` or `TensorType`, *optional*, defaults to self.return_tensors):
                Type of tensors to return. If `None`, will return the list of images.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        if 'pad_and_return_pixel_mask' in kwargs:
            logger.warning_once('The `pad_and_return_pixel_mask` argument is deprecated and will be removed in a future version, use `do_pad` instead.')
            do_pad = kwargs.pop('pad_and_return_pixel_mask')
        max_size = None
        if 'max_size' in kwargs:
            logger.warning_once("The `max_size` argument is deprecated and will be removed in a future version, use `size['longest_edge']` instead.")
            size = kwargs.pop('max_size')
        do_resize = self.do_resize if do_resize is None else do_resize
        size = self.size if size is None else size
        size = get_size_dict(size=size, max_size=max_size, default_to_square=False)
        resample = self.resample if resample is None else resample
        do_rescale = self.do_rescale if do_rescale is None else do_rescale
        rescale_factor = self.rescale_factor if rescale_factor is None else rescale_factor
        do_normalize = self.do_normalize if do_normalize is None else do_normalize
        image_mean = self.image_mean if image_mean is None else image_mean
        image_std = self.image_std if image_std is None else image_std
        do_convert_annotations = self.do_convert_annotations if do_convert_annotations is None else do_convert_annotations
        do_pad = self.do_pad if do_pad is None else do_pad
        format = self.format if format is None else format
        images = make_list_of_images(images)
        if not valid_images(images):
            raise ValueError('Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, torch.Tensor, tf.Tensor or jax.ndarray.')
        validate_preprocess_arguments(do_rescale=do_rescale, rescale_factor=rescale_factor, do_normalize=do_normalize, image_mean=image_mean, image_std=image_std, do_resize=do_resize, size=size, resample=resample)
        if annotations is not None and isinstance(annotations, dict):
            annotations = [annotations]
        if annotations is not None and len(images) != len(annotations):
            raise ValueError(f'The number of images ({len(images)}) and annotations ({len(annotations)}) do not match.')
        format = AnnotationFormat(format)
        if annotations is not None:
            validate_annotations(format, SUPPORTED_ANNOTATION_FORMATS, annotations)
        if masks_path is not None and format == AnnotationFormat.COCO_PANOPTIC and (not isinstance(masks_path, (pathlib.Path, str))):
            raise ValueError(f'The path to the directory containing the mask PNG files should be provided as a `pathlib.Path` or string object, but is {type(masks_path)} instead.')
        images = [to_numpy_array(image) for image in images]
        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once('It looks like you are trying to rescale already rescaled images. If the input images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again.')
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])
        if annotations is not None:
            prepared_images = []
            prepared_annotations = []
            for image, target in zip(images, annotations):
                target = self.prepare_annotation(image, target, format, return_segmentation_masks=return_segmentation_masks, masks_path=masks_path, input_data_format=input_data_format)
                prepared_images.append(image)
                prepared_annotations.append(target)
            images = prepared_images
            annotations = prepared_annotations
            del prepared_images, prepared_annotations
        if do_resize:
            if annotations is not None:
                resized_images, resized_annotations = ([], [])
                for image, target in zip(images, annotations):
                    orig_size = get_image_size(image, input_data_format)
                    resized_image = self.resize(image, size=size, max_size=max_size, resample=resample, input_data_format=input_data_format)
                    resized_annotation = self.resize_annotation(target, orig_size, get_image_size(resized_image, input_data_format))
                    resized_images.append(resized_image)
                    resized_annotations.append(resized_annotation)
                images = resized_images
                annotations = resized_annotations
                del resized_images, resized_annotations
            else:
                images = [self.resize(image, size=size, resample=resample, input_data_format=input_data_format) for image in images]
        if do_rescale:
            images = [self.rescale(image, rescale_factor, input_data_format=input_data_format) for image in images]
        if do_normalize:
            images = [self.normalize(image, image_mean, image_std, input_data_format=input_data_format) for image in images]
        if do_convert_annotations and annotations is not None:
            annotations = [self.normalize_annotation(annotation, get_image_size(image, input_data_format)) for annotation, image in zip(annotations, images)]
        if do_pad:
            encoded_inputs = self.pad(images, annotations=annotations, return_pixel_mask=True, data_format=data_format, input_data_format=input_data_format, return_tensors=return_tensors, update_bboxes=do_convert_annotations)
        else:
            images = [to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images]
            encoded_inputs = BatchFeature(data={'pixel_values': images}, tensor_type=return_tensors)
            if annotations is not None:
                encoded_inputs['labels'] = [BatchFeature(annotation, tensor_type=return_tensors) for annotation in annotations]
        return encoded_inputs

    def post_process(self, outputs, target_sizes):
        """
        Converts the raw output of [`DeformableDetrForObjectDetection`] into final bounding boxes in (top_left_x,
        top_left_y, bottom_right_x, bottom_right_y) format. Only supports PyTorch.

        Args:
            outputs ([`DeformableDetrObjectDetectionOutput`]):
                Raw outputs of the model.
            target_sizes (`torch.Tensor` of shape `(batch_size, 2)`):
                Tensor containing the size (height, width) of each image of the batch. For evaluation, this must be the
                original image size (before any data augmentation). For visualization, this should be the image size
                after data augment, but before padding.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        logger.warning_once('`post_process` is deprecated and will be removed in v5 of Transformers, please use `post_process_object_detection` instead, with `threshold=0.` for equivalent results.')
        out_logits, out_bbox = (outputs.logits, outputs.pred_boxes)
        if len(out_logits) != len(target_sizes):
            raise ValueError('Make sure that you pass in as many target sizes as the batch dimension of the logits')
        if target_sizes.shape[1] != 2:
            raise ValueError('Each element of target_sizes must contain the size (h, w) of each image of the batch')
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, out_logits.shape[2], rounding_mode='floor')
        labels = topk_indexes % out_logits.shape[2]
        boxes = center_to_corners_format(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results

    def post_process_object_detection(self, outputs, threshold: float=0.5, target_sizes: Union[TensorType, List[Tuple]]=None, top_k: int=100):
        """
        Converts the raw output of [`DeformableDetrForObjectDetection`] into final bounding boxes in (top_left_x,
        top_left_y, bottom_right_x, bottom_right_y) format. Only supports PyTorch.

        Args:
            outputs ([`DetrObjectDetectionOutput`]):
                Raw outputs of the model.
            threshold (`float`, *optional*):
                Score threshold to keep object detection predictions.
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                (height, width) of each image in the batch. If left to None, predictions will not be resized.
            top_k (`int`, *optional*, defaults to 100):
                Keep only top k bounding boxes before filtering by thresholding.

        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        out_logits, out_bbox = (outputs.logits, outputs.pred_boxes)
        if target_sizes is not None:
            if len(out_logits) != len(target_sizes):
                raise ValueError('Make sure that you pass in as many target sizes as the batch dimension of the logits')
        prob = out_logits.sigmoid()
        prob = prob.view(out_logits.shape[0], -1)
        k_value = min(top_k, prob.size(1))
        topk_values, topk_indexes = torch.topk(prob, k_value, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, out_logits.shape[2], rounding_mode='floor')
        labels = topk_indexes % out_logits.shape[2]
        boxes = center_to_corners_format(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        if target_sizes is not None:
            if isinstance(target_sizes, List):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                img_h, img_w = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]
        results = []
        for s, l, b in zip(scores, labels, boxes):
            score = s[s > threshold]
            label = l[s > threshold]
            box = b[s > threshold]
            results.append({'scores': score, 'labels': label, 'boxes': box})
        return results