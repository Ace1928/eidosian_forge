import math
from copy import deepcopy
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import convert_to_rgb, pad, resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import (
class SamImageProcessor(BaseImageProcessor):
    """
    Constructs a SAM image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"longest_edge": 1024}`):
            Size of the output image after resizing. Resizes the longest edge of the image to match
            `size["longest_edge"]` while maintaining the aspect ratio. Can be overridden by the `size` parameter in the
            `preprocess` method.
        mask_size (`dict`, *optional*, defaults to `{"longest_edge": 256}`):
            Size of the output segmentation map after resizing. Resizes the longest edge of the image to match
            `size["longest_edge"]` while maintaining the aspect ratio. Can be overridden by the `mask_size` parameter
            in the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Wwhether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Only has an effect if `do_rescale` is set to `True`. Can be
            overridden by the `rescale_factor` parameter in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to pad the image to the specified `pad_size`. Can be overridden by the `do_pad` parameter in the
            `preprocess` method.
        pad_size (`dict`, *optional*, defaults to `{"height": 1024, "width": 1024}`):
            Size of the output image after padding. Can be overridden by the `pad_size` parameter in the `preprocess`
            method.
        mask_pad_size (`dict`, *optional*, defaults to `{"height": 256, "width": 256}`):
            Size of the output segmentation map after padding. Can be overridden by the `mask_pad_size` parameter in
            the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """
    model_input_names = ['pixel_values']

    def __init__(self, do_resize: bool=True, size: Dict[str, int]=None, mask_size: Dict[str, int]=None, resample: PILImageResampling=PILImageResampling.BILINEAR, do_rescale: bool=True, rescale_factor: Union[int, float]=1 / 255, do_normalize: bool=True, image_mean: Optional[Union[float, List[float]]]=None, image_std: Optional[Union[float, List[float]]]=None, do_pad: bool=True, pad_size: int=None, mask_pad_size: int=None, do_convert_rgb: bool=True, **kwargs) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {'longest_edge': 1024}
        size = get_size_dict(max_size=size, default_to_square=False) if not isinstance(size, dict) else size
        pad_size = pad_size if pad_size is not None else {'height': 1024, 'width': 1024}
        pad_size = get_size_dict(pad_size, default_to_square=True)
        mask_size = mask_size if mask_size is not None else {'longest_edge': 256}
        mask_size = get_size_dict(max_size=mask_size, default_to_square=False) if not isinstance(mask_size, dict) else mask_size
        mask_pad_size = mask_pad_size if mask_pad_size is not None else {'height': 256, 'width': 256}
        mask_pad_size = get_size_dict(mask_pad_size, default_to_square=True)
        self.do_resize = do_resize
        self.size = size
        self.mask_size = mask_size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.do_pad = do_pad
        self.pad_size = pad_size
        self.mask_pad_size = mask_pad_size
        self.do_convert_rgb = do_convert_rgb

    def pad_image(self, image: np.ndarray, pad_size: Dict[str, int], data_format: Optional[Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> np.ndarray:
        """
        Pad an image to `(pad_size["height"], pad_size["width"])` with zeros to the right and bottom.

        Args:
            image (`np.ndarray`):
                Image to pad.
            pad_size (`Dict[str, int]`):
                Size of the output image after padding.
            data_format (`str` or `ChannelDimension`, *optional*):
                The data format of the image. Can be either "channels_first" or "channels_last". If `None`, the
                `data_format` of the `image` will be used.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        output_height, output_width = (pad_size['height'], pad_size['width'])
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)
        pad_width = output_width - input_width
        pad_height = output_height - input_height
        padded_image = pad(image, ((0, pad_height), (0, pad_width)), data_format=data_format, input_data_format=input_data_format, **kwargs)
        return padded_image

    def _get_preprocess_shape(self, old_shape: Tuple[int, int], longest_edge: int):
        """
        Compute the output size given input size and target long side length.
        """
        oldh, oldw = old_shape
        scale = longest_edge * 1.0 / max(oldh, oldw)
        newh, neww = (oldh * scale, oldw * scale)
        newh = int(newh + 0.5)
        neww = int(neww + 0.5)
        return (newh, neww)

    def resize(self, image: np.ndarray, size: Dict[str, int], resample: PILImageResampling=PILImageResampling.BICUBIC, data_format: Optional[Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> np.ndarray:
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary in the format `{"longest_edge": int}` specifying the size of the output image. The longest
                edge of the image will be resized to the specified size, while the other edge will be resized to
                maintain the aspect ratio.
            resample:
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.

        Returns:
            `np.ndarray`: The resized image.
        """
        size = get_size_dict(size)
        if 'longest_edge' not in size:
            raise ValueError(f'The `size` dictionary must contain the key `longest_edge`. Got {size.keys()}')
        input_size = get_image_size(image, channel_dim=input_data_format)
        output_height, output_width = self._get_preprocess_shape(input_size, size['longest_edge'])
        return resize(image, size=(output_height, output_width), resample=resample, data_format=data_format, input_data_format=input_data_format, **kwargs)

    def _preprocess(self, image: ImageInput, do_resize: bool, do_rescale: bool, do_normalize: bool, size: Optional[Dict[str, int]]=None, resample: PILImageResampling=None, rescale_factor: Optional[float]=None, image_mean: Optional[Union[float, List[float]]]=None, image_std: Optional[Union[float, List[float]]]=None, do_pad: Optional[bool]=None, pad_size: Optional[Dict[str, int]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None):
        if do_resize:
            image = self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)
        reshaped_input_size = get_image_size(image, channel_dim=input_data_format)
        if do_rescale:
            image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
        if do_normalize:
            image = self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)
        if do_pad:
            image = self.pad_image(image=image, pad_size=pad_size, input_data_format=input_data_format)
        return (image, reshaped_input_size)

    def _preprocess_image(self, image: ImageInput, do_resize: Optional[bool]=None, size: Dict[str, int]=None, resample: PILImageResampling=None, do_rescale: bool=None, rescale_factor: Optional[float]=None, do_normalize: Optional[bool]=None, image_mean: Optional[Union[float, List[float]]]=None, image_std: Optional[Union[float, List[float]]]=None, do_pad: Optional[bool]=None, pad_size: Optional[Dict[str, int]]=None, do_convert_rgb: Optional[bool]=None, data_format: Optional[Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        image = to_numpy_array(image)
        if do_convert_rgb:
            image = convert_to_rgb(image)
        image = to_numpy_array(image)
        if is_scaled_image(image) and do_rescale:
            logger.warning_once('It looks like you are trying to rescale already rescaled images. If the input images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again.')
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)
        original_size = get_image_size(image, channel_dim=input_data_format)
        image, reshaped_input_size = self._preprocess(image=image, do_resize=do_resize, size=size, resample=resample, do_rescale=do_rescale, rescale_factor=rescale_factor, do_normalize=do_normalize, image_mean=image_mean, image_std=image_std, do_pad=do_pad, pad_size=pad_size, input_data_format=input_data_format)
        if data_format is not None:
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
        return (image, original_size, reshaped_input_size)

    def _preprocess_mask(self, segmentation_map: ImageInput, do_resize: Optional[bool]=None, mask_size: Dict[str, int]=None, do_pad: Optional[bool]=None, mask_pad_size: Optional[Dict[str, int]]=None, input_data_format: Optional[Union[str, ChannelDimension]]=None) -> np.ndarray:
        segmentation_map = to_numpy_array(segmentation_map)
        if segmentation_map.ndim == 2:
            added_channel_dim = True
            segmentation_map = segmentation_map[None, ...]
            input_data_format = ChannelDimension.FIRST
        else:
            added_channel_dim = False
            if input_data_format is None:
                input_data_format = infer_channel_dimension_format(segmentation_map, num_channels=1)
        original_size = get_image_size(segmentation_map, channel_dim=input_data_format)
        segmentation_map, _ = self._preprocess(image=segmentation_map, do_resize=do_resize, size=mask_size, resample=PILImageResampling.NEAREST, do_rescale=False, do_normalize=False, do_pad=do_pad, pad_size=mask_pad_size, input_data_format=input_data_format)
        if added_channel_dim:
            segmentation_map = segmentation_map.squeeze(0)
        segmentation_map = segmentation_map.astype(np.int64)
        return (segmentation_map, original_size)

    def preprocess(self, images: ImageInput, segmentation_maps: Optional[ImageInput]=None, do_resize: Optional[bool]=None, size: Optional[Dict[str, int]]=None, mask_size: Optional[Dict[str, int]]=None, resample: Optional['PILImageResampling']=None, do_rescale: Optional[bool]=None, rescale_factor: Optional[Union[int, float]]=None, do_normalize: Optional[bool]=None, image_mean: Optional[Union[float, List[float]]]=None, image_std: Optional[Union[float, List[float]]]=None, do_pad: Optional[bool]=None, pad_size: Optional[Dict[str, int]]=None, mask_pad_size: Optional[Dict[str, int]]=None, do_convert_rgb: Optional[bool]=None, return_tensors: Optional[Union[str, TensorType]]=None, data_format: ChannelDimension=ChannelDimension.FIRST, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs):
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            segmentation_maps (`ImageInput`, *optional*):
                Segmentation map to preprocess.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Controls the size of the image after `resize`. The longest edge of the image is resized to
                `size["longest_edge"]` whilst preserving the aspect ratio.
            mask_size (`Dict[str, int]`, *optional*, defaults to `self.mask_size`):
                Controls the size of the segmentation map after `resize`. The longest edge of the image is resized to
                `size["longest_edge"]` whilst preserving the aspect ratio.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image pixel values by rescaling factor.
            rescale_factor (`int` or `float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to apply to the image pixel values.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to normalize the image by if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to normalize the image by if `do_normalize` is set to `True`.
            do_pad (`bool`, *optional*, defaults to `self.do_pad`):
                Whether to pad the image.
            pad_size (`Dict[str, int]`, *optional*, defaults to `self.pad_size`):
                Controls the size of the padding applied to the image. The image is padded to `pad_size["height"]` and
                `pad_size["width"]` if `do_pad` is set to `True`.
            mask_pad_size (`Dict[str, int]`, *optional*, defaults to `self.mask_pad_size`):
                Controls the size of the padding applied to the segmentation map. The image is padded to
                `mask_pad_size["height"]` and `mask_pad_size["width"]` if `do_pad` is set to `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
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
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(max_size=size, default_to_square=False) if not isinstance(size, dict) else size
        mask_size = mask_size if mask_size is not None else self.mask_size
        mask_size = get_size_dict(max_size=mask_size, default_to_square=False) if not isinstance(mask_size, dict) else mask_size
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_pad = do_pad if do_pad is not None else self.do_pad
        pad_size = pad_size if pad_size is not None else self.pad_size
        pad_size = get_size_dict(pad_size, default_to_square=True)
        mask_pad_size = mask_pad_size if mask_pad_size is not None else self.mask_pad_size
        mask_pad_size = get_size_dict(mask_pad_size, default_to_square=True)
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        images = make_list_of_images(images)
        if not valid_images(images):
            raise ValueError('Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, torch.Tensor, tf.Tensor or jax.ndarray.')
        if segmentation_maps is not None:
            segmentation_maps = make_list_of_images(segmentation_maps, expected_ndims=2)
            if not valid_images(segmentation_maps):
                raise ValueError('Invalid segmentation map type. Must be of type PIL.Image.Image, numpy.ndarray, torch.Tensor, tf.Tensor or jax.ndarray.')
        validate_preprocess_arguments(do_rescale=do_rescale, rescale_factor=rescale_factor, do_normalize=do_normalize, image_mean=image_mean, image_std=image_std, do_pad=do_pad, size_divisibility=pad_size, do_resize=do_resize, size=size, resample=resample)
        images, original_sizes, reshaped_input_sizes = zip(*(self._preprocess_image(image=img, do_resize=do_resize, size=size, resample=resample, do_rescale=do_rescale, rescale_factor=rescale_factor, do_normalize=do_normalize, image_mean=image_mean, image_std=image_std, do_pad=do_pad, pad_size=pad_size, do_convert_rgb=do_convert_rgb, data_format=data_format, input_data_format=input_data_format) for img in images))
        data = {'pixel_values': images, 'original_sizes': original_sizes, 'reshaped_input_sizes': reshaped_input_sizes}
        if segmentation_maps is not None:
            segmentation_maps, original_mask_sizes = zip(*(self._preprocess_mask(segmentation_map=mask, do_resize=do_resize, mask_size=mask_size, do_pad=do_pad, mask_pad_size=mask_pad_size, input_data_format=input_data_format) for mask in segmentation_maps))
            assert all((original_im_size == original_mask_size for original_im_size, original_mask_size in zip(original_sizes, original_mask_sizes))), 'Segmentation maps should be the same size as input images.'
            data['labels'] = segmentation_maps
        return BatchFeature(data=data, tensor_type=return_tensors)

    def post_process_masks(self, masks, original_sizes, reshaped_input_sizes, mask_threshold=0.0, binarize=True, pad_size=None, return_tensors='pt'):
        """
        Remove padding and upscale masks to the original image size.

        Args:
            masks (`Union[List[torch.Tensor], List[np.ndarray], List[tf.Tensor]]`):
                Batched masks from the mask_decoder in (batch_size, num_channels, height, width) format.
            original_sizes (`Union[torch.Tensor, tf.Tensor, List[Tuple[int,int]]]`):
                The original sizes of each image before it was resized to the model's expected input shape, in (height,
                width) format.
            reshaped_input_sizes (`Union[torch.Tensor, tf.Tensor, List[Tuple[int,int]]]`):
                The size of each image as it is fed to the model, in (height, width) format. Used to remove padding.
            mask_threshold (`float`, *optional*, defaults to 0.0):
                The threshold to use for binarizing the masks.
            binarize (`bool`, *optional*, defaults to `True`):
                Whether to binarize the masks.
            pad_size (`int`, *optional*, defaults to `self.pad_size`):
                The target size the images were padded to before being passed to the model. If None, the target size is
                assumed to be the processor's `pad_size`.
            return_tensors (`str`, *optional*, defaults to `"pt"`):
                If `"pt"`, return PyTorch tensors. If `"tf"`, return TensorFlow tensors.
        Returns:
            (`Union[torch.Tensor, tf.Tensor]`): Batched masks in batch_size, num_channels, height, width) format, where
            (height, width) is given by original_size.
        """
        if return_tensors == 'pt':
            return self._post_process_masks_pt(masks=masks, original_sizes=original_sizes, reshaped_input_sizes=reshaped_input_sizes, mask_threshold=mask_threshold, binarize=binarize, pad_size=pad_size)
        elif return_tensors == 'tf':
            return self._post_process_masks_tf(masks=masks, original_sizes=original_sizes, reshaped_input_sizes=reshaped_input_sizes, mask_threshold=mask_threshold, binarize=binarize, pad_size=pad_size)
        else:
            raise ValueError("return_tensors must be either 'pt' or 'tf'")

    def _post_process_masks_pt(self, masks, original_sizes, reshaped_input_sizes, mask_threshold=0.0, binarize=True, pad_size=None):
        """
        Remove padding and upscale masks to the original image size.

        Args:
            masks (`Union[List[torch.Tensor], List[np.ndarray]]`):
                Batched masks from the mask_decoder in (batch_size, num_channels, height, width) format.
            original_sizes (`Union[torch.Tensor, List[Tuple[int,int]]]`):
                The original sizes of each image before it was resized to the model's expected input shape, in (height,
                width) format.
            reshaped_input_sizes (`Union[torch.Tensor, List[Tuple[int,int]]]`):
                The size of each image as it is fed to the model, in (height, width) format. Used to remove padding.
            mask_threshold (`float`, *optional*, defaults to 0.0):
                The threshold to use for binarizing the masks.
            binarize (`bool`, *optional*, defaults to `True`):
                Whether to binarize the masks.
            pad_size (`int`, *optional*, defaults to `self.pad_size`):
                The target size the images were padded to before being passed to the model. If None, the target size is
                assumed to be the processor's `pad_size`.
        Returns:
            (`torch.Tensor`): Batched masks in batch_size, num_channels, height, width) format, where (height, width)
            is given by original_size.
        """
        requires_backends(self, ['torch'])
        pad_size = self.pad_size if pad_size is None else pad_size
        target_image_size = (pad_size['height'], pad_size['width'])
        if isinstance(original_sizes, (torch.Tensor, np.ndarray)):
            original_sizes = original_sizes.tolist()
        if isinstance(reshaped_input_sizes, (torch.Tensor, np.ndarray)):
            reshaped_input_sizes = reshaped_input_sizes.tolist()
        output_masks = []
        for i, original_size in enumerate(original_sizes):
            if isinstance(masks[i], np.ndarray):
                masks[i] = torch.from_numpy(masks[i])
            elif not isinstance(masks[i], torch.Tensor):
                raise ValueError('Input masks should be a list of `torch.tensors` or a list of `np.ndarray`')
            interpolated_mask = F.interpolate(masks[i], target_image_size, mode='bilinear', align_corners=False)
            interpolated_mask = interpolated_mask[..., :reshaped_input_sizes[i][0], :reshaped_input_sizes[i][1]]
            interpolated_mask = F.interpolate(interpolated_mask, original_size, mode='bilinear', align_corners=False)
            if binarize:
                interpolated_mask = interpolated_mask > mask_threshold
            output_masks.append(interpolated_mask)
        return output_masks

    def _post_process_masks_tf(self, masks, original_sizes, reshaped_input_sizes, mask_threshold=0.0, binarize=True, pad_size=None):
        """
        Remove padding and upscale masks to the original image size.

        Args:
            masks (`tf.Tensor`):
                Batched masks from the mask_decoder in (batch_size, num_channels, height, width) format.
            original_sizes (`tf.Tensor`):
                The original size of the images before resizing for input to the model, in (height, width) format.
            reshaped_input_sizes (`tf.Tensor`):
                The size of the image input to the model, in (height, width) format. Used to remove padding.
            mask_threshold (`float`, *optional*, defaults to 0.0):
                The threshold to use for binarizing the masks.
            binarize (`bool`, *optional*, defaults to `True`):
                Whether to binarize the masks.
            pad_size (`int`, *optional*, defaults to `self.pad_size`):
                The target size the images were padded to before being passed to the model. If None, the target size is
                assumed to be the processor's `pad_size`.
        Returns:
            (`tf.Tensor`): Batched masks in batch_size, num_channels, height, width) format, where (height, width) is
            given by original_size.
        """
        requires_backends(self, ['tf'])
        pad_size = self.pad_size if pad_size is None else pad_size
        target_image_size = (pad_size['height'], pad_size['width'])
        output_masks = []
        for i, original_size in enumerate(original_sizes):
            mask = tf.transpose(masks[i], perm=[0, 2, 3, 1])
            interpolated_mask = tf.image.resize(mask, target_image_size, method='bilinear')
            interpolated_mask = interpolated_mask[:, :reshaped_input_sizes[i][0], :reshaped_input_sizes[i][1], :]
            interpolated_mask = tf.image.resize(interpolated_mask, original_size, method='bilinear')
            if binarize:
                interpolated_mask = interpolated_mask > mask_threshold
            output_masks.append(tf.transpose(interpolated_mask, perm=[0, 3, 1, 2]))
        return output_masks

    def post_process_for_mask_generation(self, all_masks, all_scores, all_boxes, crops_nms_thresh, return_tensors='pt'):
        """
        Post processes mask that are generated by calling the Non Maximum Suppression algorithm on the predicted masks.

        Args:
            all_masks (`Union[List[torch.Tensor], List[tf.Tensor]]`):
                List of all predicted segmentation masks
            all_scores (`Union[List[torch.Tensor], List[tf.Tensor]]`):
                List of all predicted iou scores
            all_boxes (`Union[List[torch.Tensor], List[tf.Tensor]]`):
                List of all bounding boxes of the predicted masks
            crops_nms_thresh (`float`):
                Threshold for NMS (Non Maximum Suppression) algorithm.
            return_tensors (`str`, *optional*, defaults to `pt`):
                If `pt`, returns `torch.Tensor`. If `tf`, returns `tf.Tensor`.
        """
        if return_tensors == 'pt':
            return _postprocess_for_mg(all_masks, all_scores, all_boxes, crops_nms_thresh)
        elif return_tensors == 'tf':
            return _postprocess_for_mg_tf(all_masks, all_scores, all_boxes, crops_nms_thresh)

    def generate_crop_boxes(self, image, target_size, crop_n_layers: int=0, overlap_ratio: float=512 / 1500, points_per_crop: Optional[int]=32, crop_n_points_downscale_factor: Optional[List[int]]=1, device: Optional['torch.device']=None, input_data_format: Optional[Union[str, ChannelDimension]]=None, return_tensors: str='pt'):
        """
        Generates a list of crop boxes of different sizes. Each layer has (2**i)**2 boxes for the ith layer.

        Args:
            image (`np.array`):
                Input original image
            target_size (`int`):
                Target size of the resized image
            crop_n_layers (`int`, *optional*, defaults to 0):
                If >0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where
                each layer has 2**i_layer number of image crops.
            overlap_ratio (`float`, *optional*, defaults to 512/1500):
                Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of
                the image length. Later layers with more crops scale down this overlap.
            points_per_crop (`int`, *optional*, defaults to 32):
                Number of points to sample from each crop.
            crop_n_points_downscale_factor (`List[int]`, *optional*, defaults to 1):
                The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
            device (`torch.device`, *optional*, defaults to None):
                Device to use for the computation. If None, cpu will be used.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
            return_tensors (`str`, *optional*, defaults to `pt`):
                If `pt`, returns `torch.Tensor`. If `tf`, returns `tf.Tensor`.
        """
        crop_boxes, points_per_crop, cropped_images, input_labels = _generate_crop_boxes(image, target_size, crop_n_layers, overlap_ratio, points_per_crop, crop_n_points_downscale_factor, input_data_format)
        if return_tensors == 'pt':
            if device is None:
                device = torch.device('cpu')
            crop_boxes = torch.tensor(crop_boxes, device=device)
            points_per_crop = torch.tensor(points_per_crop, device=device)
            input_labels = torch.tensor(input_labels, device=device)
        elif return_tensors == 'tf':
            if device is not None:
                raise ValueError('device is not a supported argument when return_tensors is tf!')
            crop_boxes = tf.convert_to_tensor(crop_boxes)
            points_per_crop = tf.convert_to_tensor(points_per_crop)
            input_labels = tf.convert_to_tensor(input_labels)
        else:
            raise ValueError("return_tensors must be either 'pt' or 'tf'.")
        return (crop_boxes, points_per_crop, cropped_images, input_labels)

    def filter_masks(self, masks, iou_scores, original_size, cropped_box_image, pred_iou_thresh=0.88, stability_score_thresh=0.95, mask_threshold=0, stability_score_offset=1, return_tensors='pt'):
        """
        Filters the predicted masks by selecting only the ones that meets several criteria. The first criterion being
        that the iou scores needs to be greater than `pred_iou_thresh`. The second criterion is that the stability
        score needs to be greater than `stability_score_thresh`. The method also converts the predicted masks to
        bounding boxes and pad the predicted masks if necessary.

        Args:
            masks (`Union[torch.Tensor, tf.Tensor]`):
                Input masks.
            iou_scores (`Union[torch.Tensor, tf.Tensor]`):
                List of IoU scores.
            original_size (`Tuple[int,int]`):
                Size of the orginal image.
            cropped_box_image (`np.array`):
                The cropped image.
            pred_iou_thresh (`float`, *optional*, defaults to 0.88):
                The threshold for the iou scores.
            stability_score_thresh (`float`, *optional*, defaults to 0.95):
                The threshold for the stability score.
            mask_threshold (`float`, *optional*, defaults to 0):
                The threshold for the predicted masks.
            stability_score_offset (`float`, *optional*, defaults to 1):
                The offset for the stability score used in the `_compute_stability_score` method.
            return_tensors (`str`, *optional*, defaults to `pt`):
                If `pt`, returns `torch.Tensor`. If `tf`, returns `tf.Tensor`.
        """
        if return_tensors == 'pt':
            return self._filter_masks_pt(masks=masks, iou_scores=iou_scores, original_size=original_size, cropped_box_image=cropped_box_image, pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh, mask_threshold=mask_threshold, stability_score_offset=stability_score_offset)
        elif return_tensors == 'tf':
            return self._filter_masks_tf(masks=masks, iou_scores=iou_scores, original_size=original_size, cropped_box_image=cropped_box_image, pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh, mask_threshold=mask_threshold, stability_score_offset=stability_score_offset)

    def _filter_masks_pt(self, masks, iou_scores, original_size, cropped_box_image, pred_iou_thresh=0.88, stability_score_thresh=0.95, mask_threshold=0, stability_score_offset=1):
        """
        Filters the predicted masks by selecting only the ones that meets several criteria. The first criterion being
        that the iou scores needs to be greater than `pred_iou_thresh`. The second criterion is that the stability
        score needs to be greater than `stability_score_thresh`. The method also converts the predicted masks to
        bounding boxes and pad the predicted masks if necessary.

        Args:
            masks (`torch.Tensor`):
                Input masks.
            iou_scores (`torch.Tensor`):
                List of IoU scores.
            original_size (`Tuple[int,int]`):
                Size of the orginal image.
            cropped_box_image (`np.array`):
                The cropped image.
            pred_iou_thresh (`float`, *optional*, defaults to 0.88):
                The threshold for the iou scores.
            stability_score_thresh (`float`, *optional*, defaults to 0.95):
                The threshold for the stability score.
            mask_threshold (`float`, *optional*, defaults to 0):
                The threshold for the predicted masks.
            stability_score_offset (`float`, *optional*, defaults to 1):
                The offset for the stability score used in the `_compute_stability_score` method.

        """
        requires_backends(self, ['torch'])
        original_height, original_width = original_size
        iou_scores = iou_scores.flatten(0, 1)
        masks = masks.flatten(0, 1)
        if masks.shape[0] != iou_scores.shape[0]:
            raise ValueError('masks and iou_scores must have the same batch size.')
        if masks.device != iou_scores.device:
            iou_scores = iou_scores.to(masks.device)
        batch_size = masks.shape[0]
        keep_mask = torch.ones(batch_size, dtype=torch.bool, device=masks.device)
        if pred_iou_thresh > 0.0:
            keep_mask = keep_mask & (iou_scores > pred_iou_thresh)
        if stability_score_thresh > 0.0:
            stability_scores = _compute_stability_score_pt(masks, mask_threshold, stability_score_offset)
            keep_mask = keep_mask & (stability_scores > stability_score_thresh)
        scores = iou_scores[keep_mask]
        masks = masks[keep_mask]
        masks = masks > mask_threshold
        converted_boxes = _batched_mask_to_box(masks)
        keep_mask = ~_is_box_near_crop_edge(converted_boxes, cropped_box_image, [0, 0, original_width, original_height])
        scores = scores[keep_mask]
        masks = masks[keep_mask]
        converted_boxes = converted_boxes[keep_mask]
        masks = _pad_masks(masks, cropped_box_image, original_height, original_width)
        masks = _mask_to_rle_pytorch(masks)
        return (masks, scores, converted_boxes)

    def _filter_masks_tf(self, masks, iou_scores, original_size, cropped_box_image, pred_iou_thresh=0.88, stability_score_thresh=0.95, mask_threshold=0, stability_score_offset=1):
        """
        Filters the predicted masks by selecting only the ones that meets several criteria. The first criterion being
        that the iou scores needs to be greater than `pred_iou_thresh`. The second criterion is that the stability
        score needs to be greater than `stability_score_thresh`. The method also converts the predicted masks to
        bounding boxes and pad the predicted masks if necessary.

        Args:
            masks (`tf.Tensor`):
                Input masks.
            iou_scores (`tf.Tensor`):
                List of IoU scores.
            original_size (`Tuple[int,int]`):
                Size of the orginal image.
            cropped_box_image (`np.array`):
                The cropped image.
            pred_iou_thresh (`float`, *optional*, defaults to 0.88):
                The threshold for the iou scores.
            stability_score_thresh (`float`, *optional*, defaults to 0.95):
                The threshold for the stability score.
            mask_threshold (`float`, *optional*, defaults to 0):
                The threshold for the predicted masks.
            stability_score_offset (`float`, *optional*, defaults to 1):
                The offset for the stability score used in the `_compute_stability_score` method.

        """
        requires_backends(self, ['tf'])
        original_height, original_width = original_size
        iou_scores = tf.reshape(iou_scores, [iou_scores.shape[0] * iou_scores.shape[1], iou_scores.shape[2:]])
        masks = tf.reshape(masks, [masks.shape[0] * masks.shape[1], masks.shape[2:]])
        if masks.shape[0] != iou_scores.shape[0]:
            raise ValueError('masks and iou_scores must have the same batch size.')
        batch_size = masks.shape[0]
        keep_mask = tf.ones(batch_size, dtype=tf.bool)
        if pred_iou_thresh > 0.0:
            keep_mask = keep_mask & (iou_scores > pred_iou_thresh)
        if stability_score_thresh > 0.0:
            stability_scores = _compute_stability_score_tf(masks, mask_threshold, stability_score_offset)
            keep_mask = keep_mask & (stability_scores > stability_score_thresh)
        scores = iou_scores[keep_mask]
        masks = masks[keep_mask]
        masks = masks > mask_threshold
        converted_boxes = _batched_mask_to_box_tf(masks)
        keep_mask = ~_is_box_near_crop_edge_tf(converted_boxes, cropped_box_image, [0, 0, original_width, original_height])
        scores = scores[keep_mask]
        masks = masks[keep_mask]
        converted_boxes = converted_boxes[keep_mask]
        masks = _pad_masks_tf(masks, cropped_box_image, original_height, original_width)
        masks = _mask_to_rle_tf(masks)
        return (masks, scores, converted_boxes)