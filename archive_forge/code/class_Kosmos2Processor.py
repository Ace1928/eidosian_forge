import copy
import math
import re
from typing import List, Optional, Tuple, Union
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput, is_batched
from ...processing_utils import ProcessorMixin
from ...tokenization_utils import AddedToken
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, TextInput, TruncationStrategy
from ...utils import TensorType
class Kosmos2Processor(ProcessorMixin):
    """
    Constructs an KOSMOS-2 processor which wraps a KOSMOS-2 image processor and a KOSMOS-2 tokenizer into a single
    processor.

    [`Kosmos2Processor`] offers all the functionalities of [`CLIPImageProcessor`] and some functionalities of
    [`XLMRobertaTokenizerFast`]. See the docstring of [`~Kosmos2Processor.__call__`] and [`~Kosmos2Processor.decode`]
    for more information.

    Args:
        image_processor (`CLIPImageProcessor`):
            An instance of [`CLIPImageProcessor`]. The image processor is a required input.
        tokenizer (`XLMRobertaTokenizerFast`):
            An instance of ['XLMRobertaTokenizerFast`]. The tokenizer is a required input.
        num_patch_index_tokens (`int`, *optional*, defaults to 1024):
            The number of tokens that represent patch indices.
    """
    attributes = ['image_processor', 'tokenizer']
    image_processor_class = 'CLIPImageProcessor'
    tokenizer_class = ('XLMRobertaTokenizer', 'XLMRobertaTokenizerFast')

    def __init__(self, image_processor, tokenizer, num_patch_index_tokens=1024):
        tokenizer.return_token_type_ids = False
        self.eod_token = '</doc>'
        self.boi_token = '<image>'
        self.eoi_token = '</image>'
        self.eoc_token = '</chunk>'
        self.eol_token = '</line>'
        self.bop_token = '<phrase>'
        self.eop_token = '</phrase>'
        self.boo_token = '<object>'
        self.eoo_token = '</object>'
        self.dom_token = '</delimiter_of_multi_objects/>'
        self.grd_token = '<grounding>'
        self.tag_tokens = [self.eod_token, self.boi_token, self.eoi_token, self.eoc_token, self.eol_token, self.bop_token, self.eop_token, self.boo_token, self.eoo_token, self.dom_token, self.grd_token]
        self.num_patch_index_tokens = num_patch_index_tokens
        patch_index_tokens = [f'<patch_index_{str(x).zfill(4)}>' for x in range(self.num_patch_index_tokens)]
        tokens_to_add = []
        for token in self.tag_tokens + patch_index_tokens:
            tokens_to_add.append(AddedToken(token, lstrip=True, rstrip=False, normalized=False))
        tokenizer.add_tokens(tokens_to_add)
        super().__init__(image_processor, tokenizer)

    def __call__(self, images: ImageInput=None, text: Union[TextInput, List[TextInput]]=None, bboxes: BboxInput=None, num_image_tokens: Optional[int]=64, first_image_token_id: Optional[int]=None, add_special_tokens: bool=True, add_eos_token: bool=False, padding: Union[bool, str, PaddingStrategy]=False, truncation: Union[bool, str, TruncationStrategy]=None, max_length: Optional[int]=None, pad_to_multiple_of: Optional[int]=None, return_attention_mask: Optional[bool]=None, return_length: bool=False, verbose: bool=True, return_tensors: Optional[Union[str, TensorType]]=None, **kwargs) -> BatchFeature:
        """
        This method uses [`CLIPImageProcessor.__call__`] method to prepare image(s) for the model, and
        [`XLMRobertaTokenizerFast.__call__`] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.

        The rest of this documentation shows the arguments specific to `Kosmos2Processor`.

        Args:
            bboxes (`Union[List[Tuple[int]], List[Tuple[float]], List[List[Tuple[int]]], List[List[Tuple[float]]]]`, *optional*):
                The bounding bboxes associated to `texts`.
            num_image_tokens (`int`, defaults to 64):
                The number of (consecutive) places that are used to mark the placeholders to store image information.
                This should be the same as `latent_query_num` in the instance of `Kosmos2Config` you are using.
            first_image_token_id (`int`, *optional*):
                The token id that will be used for the first place of the subsequence that is reserved to store image
                information. If unset, will default to `self.tokenizer.unk_token_id + 1`.
            add_eos_token (`bool`, defaults to `False`):
                Whether or not to include `EOS` token id in the encoding when `add_special_tokens=True`.
        """
        if images is None and text is None:
            raise ValueError('You have to specify either images or text.')
        encoding = BatchFeature()
        if images is not None:
            image_encoding = self.image_processor(images, return_tensors=return_tensors)
            encoding.update(image_encoding)
        if text is not None:
            text = self.preprocess_examples(text, images, bboxes, num_image_tokens=num_image_tokens)
            if add_special_tokens and (not add_eos_token):
                if isinstance(text, str):
                    text = f'{self.tokenizer.bos_token}{text}'
                elif isinstance(text, list):
                    text = [f'{self.tokenizer.bos_token}{s}' for s in text]
            text_encoding = self.tokenizer(text=text, add_special_tokens=add_special_tokens and add_eos_token, padding=padding and images is None, truncation=truncation, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of if images is None else pad_to_multiple_of, return_attention_mask=return_attention_mask, verbose=verbose, return_tensors=return_tensors if images is None else None, **kwargs)
            encoding.update(text_encoding)
        if text is not None and images is not None:
            if first_image_token_id is None:
                first_image_token_id = self.tokenizer.unk_token_id + 1
            with_bos = add_special_tokens
            start_index = int(with_bos) + 1
            image_token_ids = list(range(first_image_token_id, first_image_token_id + num_image_tokens))
            base_image_embeds_position_mask = [0] + [1] * num_image_tokens + [0]
            input_ids = []
            image_embeds_position_mask = []
            all_input_ids = encoding['input_ids']
            if isinstance(text, str):
                all_input_ids = [all_input_ids]
                encoding['attention_mask'] = [encoding['attention_mask']]
            for text_ids in all_input_ids:
                text_ids = text_ids[:start_index] + image_token_ids + text_ids[start_index + num_image_tokens:]
                input_ids.append(text_ids)
                mask = copy.copy(base_image_embeds_position_mask)
                if with_bos:
                    mask = [0] + mask
                mask += [0] * (len(text_ids) - len(mask))
                image_embeds_position_mask.append(mask)
            if isinstance(text, list):
                sorted_length = sorted([(idx, len(x)) for idx, x in enumerate(text_encoding.input_ids)], key=lambda x: x[-1])
                _, min_len_not_padded = sorted_length[0]
                idx, _ = sorted_length[-1]
                text_encoding = self.tokenizer(text=[text[idx]], add_special_tokens=add_special_tokens and add_eos_token, padding=padding, truncation=truncation, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of, verbose=verbose, return_tensors=None, **kwargs)
                max_len_padded = len(text_encoding.input_ids[0])
                if min_len_not_padded != max_len_padded:
                    if self.tokenizer.padding_side == 'right':
                        input_ids = [x + [self.tokenizer.pad_token_id] * (max_len_padded - len(x)) for x in input_ids]
                        image_embeds_position_mask = [x + [0] * (max_len_padded - len(x)) for x in image_embeds_position_mask]
                        encoding['attention_mask'] = [x + [0] * (max_len_padded - len(x)) for x in encoding['attention_mask']]
                    elif self.tokenizer.padding_side == 'left':
                        input_ids = [[self.tokenizer.pad_token_id] * (max_len_padded - len(x)) + x for x in input_ids]
                        image_embeds_position_mask = [[0] * (max_len_padded - len(x)) + x for x in image_embeds_position_mask]
                        encoding['attention_mask'] = [[0] * (max_len_padded - len(x)) + x for x in encoding['attention_mask']]
            if isinstance(text, str) and return_tensors is None:
                input_ids = input_ids[0]
                encoding['attention_mask'] = encoding['attention_mask'][0]
                image_embeds_position_mask = image_embeds_position_mask[0]
            encoding.update(BatchEncoding(data={'input_ids': input_ids, 'attention_mask': encoding['attention_mask'], 'image_embeds_position_mask': image_embeds_position_mask}, tensor_type=return_tensors))
        return encoding

    def _check_bboxes_for_single_text(self, bboxes):
        """
        Check `bboxes` for a single text example. It could be
            - `None`: no bounding box associated to a text.
            - A list with each element being the bounding boxes associated to one `<phrase> ... </phrase>` pair found
              in a text. This could be:
                  - `None`: no bounding box associated to a `<phrase> ... </phrase>` pair.
                  - A tuple of 2 integers: A single bounding box specified by patch indices.
                  - A tuple of 4 float point number: A single bounding box specified by (normalized) coordinates.
                  - A list containing the above 2 tuple types: Multiple bounding boxes for a
                   `<phrase> ... </phrase>` pair.
        """
        if bboxes is None:
            return
        elif not isinstance(bboxes, list):
            raise ValueError('`bboxes` (for a single text example) should be `None` or a list.')
        for bbox in bboxes:
            if bbox is None:
                continue
            elif not isinstance(bbox, list):
                bbox = [bbox]
            for element in bbox:
                if not isinstance(element, tuple) or not (len(element) == 2 and all((isinstance(x, int) for x in element)) or (len(element) == 4 and all((isinstance(x, float) for x in element)))):
                    raise ValueError('Each element in `bboxes` (for a single text example) should be either `None`, a tuple containing 2 integers or 4 float point numbers, or a list containing such tuples. Also make sure the arguments `texts` and `bboxes` passed to `preprocess_text` are both in batches or both for a single example.')

    def _preprocess_single_example(self, text, image, bboxes, img_info_tokens):
        text = text.strip()
        if image is not None:
            text = f'{img_info_tokens} {text}'
        text = self._insert_patch_index_tokens(text, bboxes)
        return text

    def preprocess_examples(self, texts: Union[TextInput, List[TextInput]], images: ImageInput=None, bboxes: BboxInput=None, num_image_tokens: Optional[int]=64) -> Union[str, List[str]]:
        """Add image and bounding box information to `texts` as image and patch index tokens.

        Args:
            texts (`Union[TextInput, List[TextInput]]`): The texts to be processed.
            images (`ImageInput`, *optional*): The images associated to `texts`.
            bboxes (`Union[List[Tuple[int]], List[Tuple[float]], List[List[Tuple[int]]], List[List[Tuple[float]]]]`, *optional*):
                The bounding bboxes associated to `texts`.
            num_image_tokens (`int`, *optional*, defaults to 64):
                The number of image tokens (used as latent queries). This should corresponds to the `latent_query_num`
                attribute in `Kosmos2Config`.

        Returns:
            `Union[TextInput, List[TextInput]]`: The processed texts with image and patch index tokens.
        """
        img_tokens = [self.boi_token] * num_image_tokens
        img_info_tokens = ' '.join([self.boi_token] + img_tokens + [self.eoi_token])
        batched = True
        if isinstance(texts, str):
            batched = False
            texts = [texts]
        if images is None:
            images = [None] * len(texts)
        elif not is_batched(images):
            images = [images]
        if len(texts) != len(images):
            raise ValueError(f'The number of examples in `texts` and `images` should be the same. Got {len(texts)} v.s. {len(images)} instead.')
        if not batched:
            self._check_bboxes_for_single_text(bboxes)
            bboxes = [bboxes]
        elif bboxes is not None:
            if not isinstance(bboxes, list):
                raise ValueError('`bboxes` should be `None` or a list (as a batch) when `texts` is passed as a batch.')
            for x in bboxes:
                self._check_bboxes_for_single_text(x)
        else:
            bboxes = [None] * len(texts)
        if len(bboxes) != len(texts):
            raise ValueError(f'The number of examples in `texts` and `bboxes` should be the same. Got {len(texts)} v.s. {len(bboxes)} instead.')
        result = [self._preprocess_single_example(text, image, bbox, img_info_tokens) for text, image, bbox in zip(texts, images, bboxes)]
        if not batched:
            result = result[0]
        return result

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_generation(self, text, cleanup_and_extract=True):
        caption = text.split(self.eoi_token)[-1]
        if cleanup_and_extract:
            return clean_text_and_extract_entities_with_bboxes(caption)
        return caption

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    def _insert_patch_index_tokens(self, text: str, bboxes: Union[List[Tuple[int]], List[Tuple[float]]]) -> str:
        if bboxes is None or len(bboxes) == 0:
            return text
        matched_phrases = list(re.finditer('<phrase>.+?</phrase>', string=text))
        if len(matched_phrases) != len(bboxes):
            raise ValueError(f'The number of elements in `bboxes` should be the same as the number of `<phrase> ... </phrase>` pairs in `text`. Got {len(matched_phrases)} v.s. {len(bboxes)} instead.')
        curr_pos = 0
        buffer = []
        for matched, bbox in zip(matched_phrases, bboxes):
            _, end = matched.span()
            buffer.append(text[curr_pos:end])
            curr_pos = end
            if bbox is None:
                continue
            if isinstance(bbox, tuple):
                bbox = [bbox]
            patch_index_strings = []
            if not all((box is not None for box in bbox)):
                raise ValueError('The multiple bounding boxes for a single phrase should not contain any `None` value.')
            for box in bbox:
                patch_index_1, patch_index_2 = self._convert_bbox_to_patch_index_tokens(box)
                patch_index_strings.append(f'{patch_index_1} {patch_index_2}')
            if len(patch_index_strings) == 0:
                continue
            position_str = ' </delimiter_of_multi_objects/> '.join(patch_index_strings)
            buffer.append(f'<object> {position_str} </object>')
        if curr_pos < len(text):
            buffer.append(text[curr_pos:])
        text = ''.join(buffer)
        return text

    def _convert_bbox_to_patch_index_tokens(self, bbox: Union[Tuple[int, int], Tuple[float, float, float, float]]) -> Tuple[str, str]:
        if len(bbox) == 2:
            idx_1, idx_2 = bbox
        else:
            num_patches_per_side = int(math.sqrt(self.num_patch_index_tokens))
            idx_1, idx_2 = coordinate_to_patch_index(bbox, num_patches_per_side)
        token_1 = f'<patch_index_{str(idx_1).zfill(4)}>'
        token_2 = f'<patch_index_{str(idx_2).zfill(4)}>'
        return (token_1, token_2)