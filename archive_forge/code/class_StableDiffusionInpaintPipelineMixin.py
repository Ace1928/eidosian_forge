import inspect
from typing import Callable, List, Optional, Union
import numpy as np
import PIL
import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils import PIL_INTERPOLATION
from .pipeline_stable_diffusion import StableDiffusionPipelineMixin
class StableDiffusionInpaintPipelineMixin(StableDiffusionPipelineMixin):

    def check_inputs(self, prompt: Union[str, List[str]], height: Optional[int], width: Optional[int], callback_steps: int, negative_prompt: Optional[str]=None, prompt_embeds: Optional[np.ndarray]=None, negative_prompt_embeds: Optional[np.ndarray]=None):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f'`height` and `width` have to be divisible by 8 but are {height} and {width}.')
        if callback_steps is None or (callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)):
            raise ValueError(f'`callback_steps` has to be a positive integer but is {callback_steps} of type {type(callback_steps)}.')
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(f'Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to only forward one of the two.')
        elif prompt is None and prompt_embeds is None:
            raise ValueError('Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined.')
        elif prompt is not None and (not isinstance(prompt, str) and (not isinstance(prompt, list))):
            raise ValueError(f'`prompt` has to be of type `str` or `list` but is {type(prompt)}')
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(f'Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to only forward one of the two.')
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(f'`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds` {negative_prompt_embeds.shape}.')

    @torch.no_grad()
    def __call__(self, prompt: Union[str, List[str]], image: PIL.Image.Image, mask_image: PIL.Image.Image, height: Optional[int]=None, width: Optional[int]=None, num_inference_steps: int=50, guidance_scale: float=7.5, negative_prompt: Optional[Union[str, List[str]]]=None, num_images_per_prompt: int=1, eta: float=0.0, generator: Optional[np.random.RandomState]=None, latents: Optional[np.ndarray]=None, prompt_embeds: Optional[np.ndarray]=None, negative_prompt_embeds: Optional[np.ndarray]=None, output_type: str='pil', return_dict: bool=True, callback: Optional[Callable[[int, int, np.ndarray], None]]=None, callback_steps: int=1):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`Union[str, List[str]]`):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`PIL.Image.Image`):
                `Image`, or tensor representing an image batch which will be upscaled.
            mask_image (`PIL.Image.Image`):
                `Image`, or tensor representing a masked image batch which will be upscaled.
            height (`Optional[int]`, defaults to None):
                The height in pixels of the generated image.
            width (`Optional[int]`, defaults to None):
                The width in pixels of the generated image.
            num_inference_steps (`int`, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`Optional[Union[str, list]]`):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. Ignored when not using guidance (i.e., ignored if `guidance_scale`
                is less than `1`).
            num_images_per_prompt (`int`, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`Optional[np.random.RandomState]`, defaults to `None`)::
                A np.random.RandomState to make generation deterministic.
            latents (`Optional[np.ndarray]`, defaults to `None`):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`Optional[np.ndarray]`, defaults to `None`):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`Optional[np.ndarray]`, defaults to `None`):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (Optional[Callable], defaults to `None`):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        height = height or self.unet.config.get('sample_size', 64) * self.vae_scale_factor
        width = width or self.unet.config.get('sample_size', 64) * self.vae_scale_factor
        self.check_inputs(prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        if generator is None:
            generator = np.random
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_embeds = self._encode_prompt(prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds)
        num_channels_latents = self.vae_decoder.config.get('latent_channels', 4)
        num_channels_unet = self.unet.config.get('in_channels', 9)
        latents_shape = (batch_size * num_images_per_prompt, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        latents_dtype = prompt_embeds.dtype
        if latents is None:
            latents = generator.randn(*latents_shape).astype(latents_dtype)
        elif latents.shape != latents_shape:
            raise ValueError(f'Unexpected latents shape, got {latents.shape}, expected {latents_shape}')
        mask, masked_image = prepare_mask_and_masked_image(image, mask_image, latents_shape[-2:], self.vae_scale_factor)
        mask = mask.astype(latents.dtype)
        masked_image = masked_image.astype(latents.dtype)
        masked_image_latents = self.vae_encoder(sample=masked_image)[0]
        scaling_factor = self.vae_decoder.config.get('scaling_factor', 0.18215)
        masked_image_latents = scaling_factor * masked_image_latents
        mask = mask.repeat(batch_size * num_images_per_prompt, 0)
        masked_image_latents = masked_image_latents.repeat(batch_size * num_images_per_prompt, 0)
        mask = np.concatenate([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = np.concatenate([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        if num_channels_unet == 9:
            num_channels_mask = mask.shape[1]
            num_channels_masked_image = masked_image_latents.shape[1]
            if num_channels_latents + num_channels_mask + num_channels_masked_image != num_channels_unet:
                raise ValueError(f'Incorrect configuration settings! The config of `pipeline.unet`: expects {num_channels_unet} but received `num_channels_latents`: {num_channels_latents} + `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image} = {num_channels_latents + num_channels_masked_image + num_channels_mask}. Please verify the config of `pipeline.unet` or your `mask_image` or `image` input.')
        elif num_channels_unet != 4:
            raise ValueError(f'The unet {self.unet.__class__} should have either 4 or 9 input channels, not {num_channels_unet}.')
        latents = latents * np.float64(self.scheduler.init_noise_sigma)
        accepts_eta = 'eta' in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs['eta'] = eta
        timestep_dtype = self.unet.input_dtype.get('timestep', np.float32)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        for i, t in enumerate(self.progress_bar(timesteps)):
            latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(torch.from_numpy(latent_model_input), t)
            latent_model_input = latent_model_input.cpu().numpy()
            if num_channels_unet == 9:
                latent_model_input = np.concatenate([latent_model_input, mask, masked_image_latents], axis=1)
            timestep = np.array([t], dtype=timestep_dtype)
            noise_pred = self.unet(sample=latent_model_input, timestep=timestep, encoder_hidden_states=prompt_embeds)[0]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            scheduler_output = self.scheduler.step(torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs)
            latents = scheduler_output.prev_sample.numpy()
            if i == len(timesteps) - 1 or (i + 1 > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
        if output_type == 'latent':
            image = latents
            has_nsfw_concept = None
        else:
            latents /= scaling_factor
            image = np.concatenate([self.vae_decoder(latent_sample=latents[i:i + 1])[0] for i in range(latents.shape[0])])
            image, has_nsfw_concept = self.run_safety_checker(image)
        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        if not return_dict:
            return (image, has_nsfw_concept)
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)