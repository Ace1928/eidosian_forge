from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils import get_from_dict_or_env
Run a saved Actor task on Apify and wait for results to be ready.
        Args:
            task_id (str): The ID or name of the task on the Apify platform.
            task_input (Dict): The input object of the task that you're trying to run.
                Overrides the task's saved input.
            dataset_mapping_function (Callable): A function that takes a single
                dictionary (an Apify dataset item) and converts it to an
                instance of the Document class.
            build (str, optional): Optionally specifies the actor build to run.
                It can be either a build tag or build number.
            memory_mbytes (int, optional): Optional memory limit for the run,
                in megabytes.
            timeout_secs (int, optional): Optional timeout for the run, in seconds.
        Returns:
            ApifyDatasetLoader: A loader that will fetch the records from the
                task run's default dataset.
        