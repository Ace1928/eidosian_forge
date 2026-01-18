import fire
import os
import sys
import time
import torch
from transformers import AutoTokenizer
from llama_recipes.inference.safety_utils import get_safety_checker
from llama_recipes.inference.model_utils import load_model, load_peft_model

        Prints the safety results for a prompt.

        Parameters:
        - are_safe_prompt (bool): Indicates whether the prompt is safe.
        - prompt (str): The prompt that was checked for safety.
        - safety_results (list of tuples): A list of tuples containing the method, safety status, and safety report.
        - prompt_type (str): The type of prompt (User/System).
        