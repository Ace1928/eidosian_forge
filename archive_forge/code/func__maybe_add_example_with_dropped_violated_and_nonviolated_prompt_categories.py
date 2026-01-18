import copy
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Sequence
def _maybe_add_example_with_dropped_violated_and_nonviolated_prompt_categories(training_example: TrainingExample, formatted_examples_being_built: List[str], indices_of_all_categories: range, violated_category_indices: List[int], nonviolated_category_indices: List[int], formatter_configs: FormatterConfigs) -> None:
    """
    Same as in _maybe_add_example_with_dropped_nonviolated_prompt_categories but we
    also drop all of the violated categories from the llama guard prompt.
    """
    if training_example.label == 'safe' or not formatter_configs.augmentation_configs.should_add_examples_with_dropped_violated_and_nonviolated_prompt_categories:
        return
    random_nonviolated_category_indices_to_drop = random.sample(nonviolated_category_indices, random.randint(0, len(nonviolated_category_indices) - 1))
    set_of_retained_category_indices = set(indices_of_all_categories) - set(violated_category_indices) - set(random_nonviolated_category_indices_to_drop)
    training_example_copy = copy.deepcopy(training_example)
    training_example_copy.label = 'safe'
    training_example_copy.violated_category_codes = []
    training_example_copy.explanation = formatter_configs.augmentation_configs.explanation_for_augmentation_with_dropped_violated_and_nonviolated_prompt_categories
    formatted_examples_being_built.append(_create_formatted_finetuning_example(training_example_copy, formatter_configs, category_indices_to_include_in_llama_guard_prompt=list(set_of_retained_category_indices)))