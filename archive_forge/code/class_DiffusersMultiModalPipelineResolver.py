import logging
from typing import Any, Dict, List, Sequence
import wandb
from wandb.sdk.integration_utils.auto_logging import Response
from .utils import (
class DiffusersMultiModalPipelineResolver:
    """Resolver for  request and responses from [HuggingFace Diffusers](https://huggingface.co/docs/diffusers/index) multi-modal Diffusion Pipelines, providing necessary data transformations, formatting, and logging.

    This resolver is internally involved in the
    `__call__` for `wandb.integration.diffusers.pipeline_resolver.DiffusersPipelineResolver`.
    This is based on `wandb.sdk.integration_utils.auto_logging.RequestResponseResolver`.

    Arguments:
        pipeline_name: (str) The name of the Diffusion Pipeline.
    """

    def __init__(self, pipeline_name: str, pipeline_call_count: int) -> None:
        self.pipeline_name = pipeline_name
        self.pipeline_call_count = pipeline_call_count
        columns = []
        if pipeline_name in SUPPORTED_MULTIMODAL_PIPELINES:
            columns += SUPPORTED_MULTIMODAL_PIPELINES[pipeline_name]['table-schema']
        else:
            wandb.Error('Pipeline not supported for logging')
        self.wandb_table = wandb.Table(columns=columns)

    def __call__(self, args: Sequence[Any], kwargs: Dict[str, Any], response: Response, start_time: float, time_elapsed: float) -> Any:
        """Main call method for the `DiffusersPipelineResolver` class.

        Arguments:
            args: (Sequence[Any]) List of arguments.
            kwargs: (Dict[str, Any]) Dictionary of keyword arguments.
            response: (wandb.sdk.integration_utils.auto_logging.Response) The response from
                the request.
            start_time: (float) Time when request started.
            time_elapsed: (float) Time elapsed for the request.

        Returns:
            Packed data as a dictionary for logging to wandb, None if an exception occurred.
        """
        try:
            pipeline, args = (args[0], args[1:])
            kwargs = get_updated_kwargs(pipeline, args, kwargs)
            pipeline_configs = dict(pipeline.config)
            pipeline_configs['pipeline-name'] = self.pipeline_name
            if 'workflow' not in wandb.config:
                wandb.config.update({'workflow': [{'pipeline': pipeline_configs, 'params': kwargs, 'stage': f'Pipeline-Call-{self.pipeline_call_count}'}]})
            else:
                existing_workflow = wandb.config.workflow
                updated_workflow = existing_workflow + [{'pipeline': pipeline_configs, 'params': kwargs, 'stage': f'Pipeline-Call-{self.pipeline_call_count}'}]
                wandb.config.update({'workflow': updated_workflow}, allow_val_change=True)
            loggable_dict = self.prepare_loggable_dict(pipeline, response, kwargs)
            return loggable_dict
        except Exception as e:
            logger.warning(e)
        return None

    def get_output_images(self, response: Response) -> List:
        """Unpack the generated images, audio, video, etc. from the Diffusion Pipeline's response.

        Arguments:
            response: (wandb.sdk.integration_utils.auto_logging.Response) The response from
                the request.

        Returns:
            List of generated images, audio, video, etc.
        """
        if 'output-type' not in SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]:
            return response.images
        elif SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]['output-type'] == 'video':
            if self.pipeline_name in ['ShapEPipeline']:
                return response.images
            return response.frames
        elif SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]['output-type'] == 'audio':
            return response.audios

    def log_media(self, image: Any, loggable_kwarg_chunks: List, idx: int) -> None:
        """Log the generated images, audio, video, etc. from the Diffusion Pipeline's response along with an optional caption to a media panel in the run.

        Arguments:
            image: (Any) The generated images, audio, video, etc. from the Diffusion
                Pipeline's response.
            loggable_kwarg_chunks: (List) Loggable chunks of kwargs.
        """
        if 'output-type' not in SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]:
            try:
                caption = ''
                if self.pipeline_name in ['StableDiffusionXLPipeline', 'StableDiffusionXLImg2ImgPipeline']:
                    prompt_index = SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]['kwarg-logging'].index('prompt')
                    prompt2_index = SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]['kwarg-logging'].index('prompt_2')
                    caption = f'Prompt-1: {loggable_kwarg_chunks[prompt_index][idx]}\nPrompt-2: {loggable_kwarg_chunks[prompt2_index][idx]}'
                else:
                    prompt_index = SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]['kwarg-logging'].index('prompt')
                    caption = loggable_kwarg_chunks[prompt_index][idx]
            except ValueError:
                caption = None
            wandb.log({f'Generated-Image/Pipeline-Call-{self.pipeline_call_count}': wandb.Image(image, caption=caption)})
        elif SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]['output-type'] == 'video':
            try:
                prompt_index = SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]['kwarg-logging'].index('prompt')
                caption = loggable_kwarg_chunks[prompt_index][idx]
            except ValueError:
                caption = None
            wandb.log({f'Generated-Video/Pipeline-Call-{self.pipeline_call_count}': wandb.Video(postprocess_pils_to_np(image), fps=4, caption=caption)})
        elif SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]['output-type'] == 'audio':
            try:
                prompt_index = SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]['kwarg-logging'].index('prompt')
                caption = loggable_kwarg_chunks[prompt_index][idx]
            except ValueError:
                caption = None
            wandb.log({f'Generated-Audio/Pipeline-Call-{self.pipeline_call_count}': wandb.Audio(image, sample_rate=16000, caption=caption)})

    def add_data_to_table(self, image: Any, loggable_kwarg_chunks: List, idx: int) -> None:
        """Populate the row of the `wandb.Table`.

        Arguments:
            image: (Any) The generated images, audio, video, etc. from the Diffusion
                Pipeline's response.
            loggable_kwarg_chunks: (List) Loggable chunks of kwargs.
            idx: (int) Chunk index.
        """
        table_row = []
        kwarg_actions = SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]['kwarg-actions']
        for column_idx, loggable_kwarg_chunk in enumerate(loggable_kwarg_chunks):
            if kwarg_actions[column_idx] is None:
                table_row.append(loggable_kwarg_chunk[idx] if loggable_kwarg_chunk[idx] is not None else '')
            else:
                table_row.append(kwarg_actions[column_idx](loggable_kwarg_chunk[idx]))
        if 'output-type' not in SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]:
            table_row.append(wandb.Image(image))
        elif SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]['output-type'] == 'video':
            table_row.append(wandb.Video(postprocess_pils_to_np(image), fps=4))
        elif SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]['output-type'] == 'audio':
            table_row.append(wandb.Audio(image, sample_rate=16000))
        self.wandb_table.add_data(*table_row)

    def prepare_loggable_dict(self, pipeline: Any, response: Response, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare the loggable dictionary, which is the packed data as a dictionary for logging to wandb, None if an exception occurred.

        Arguments:
            pipeline: (Any) The Diffusion Pipeline.
            response: (wandb.sdk.integration_utils.auto_logging.Response) The response from
                the request.
            kwargs: (Dict[str, Any]) Dictionary of keyword arguments.

        Returns:
            Packed data as a dictionary for logging to wandb, None if an exception occurred.
        """
        images = self.get_output_images(response)
        if self.pipeline_name == 'StableDiffusionXLPipeline' and kwargs['output_type'] == 'latent':
            images = decode_sdxl_t2i_latents(pipeline, response.images)
        if self.pipeline_name in ['TextToVideoSDPipeline', 'TextToVideoZeroPipeline']:
            video = postprocess_np_arrays_for_video(images, normalize=self.pipeline_name == 'TextToVideoZeroPipeline')
            wandb.log({f'Generated-Video/Pipeline-Call-{self.pipeline_call_count}': wandb.Video(video, fps=4, caption=kwargs['prompt'])})
            loggable_kwarg_ids = SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]['kwarg-logging']
            table_row = [kwargs[loggable_kwarg_ids[idx]] for idx in range(len(loggable_kwarg_ids))]
            table_row.append(wandb.Video(video, fps=4))
            self.wandb_table.add_data(*table_row)
        else:
            loggable_kwarg_ids = SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]['kwarg-logging']
            loggable_kwarg_chunks = []
            for loggable_kwarg_id in loggable_kwarg_ids:
                loggable_kwarg_chunks.append(kwargs[loggable_kwarg_id] if isinstance(kwargs[loggable_kwarg_id], list) else [kwargs[loggable_kwarg_id]])
            images = chunkify(images, len(loggable_kwarg_chunks[0]))
            for idx in range(len(loggable_kwarg_chunks[0])):
                for image in images[idx]:
                    self.log_media(image, loggable_kwarg_chunks, idx)
                    self.add_data_to_table(image, loggable_kwarg_chunks, idx)
        return {f'Result-Table/Pipeline-Call-{self.pipeline_call_count}': self.wandb_table}