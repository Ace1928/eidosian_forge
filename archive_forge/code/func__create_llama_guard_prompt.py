import copy
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Sequence
def _create_llama_guard_prompt(training_example: TrainingExample, category_indices_to_include: List[int], formatter_configs: FormatterConfigs) -> str:
    full_guidelines_text = ''
    for rewritten_category_index_for_current_prompt, original_category_index in enumerate(category_indices_to_include):
        category = formatter_configs.guidelines.categories[original_category_index]
        newline_for_every_category_after_first = f'\n' if rewritten_category_index_for_current_prompt > 0 else ''
        full_guidelines_text += f'{newline_for_every_category_after_first}{formatter_configs.guidelines.category_code_prefix}{rewritten_category_index_for_current_prompt + 1}: {category.name}. '
        if formatter_configs.llama_guard_prompt_configs.should_include_category_descriptions:
            full_guidelines_text += f'\n{category.description}'
    conversation = {'human': training_example.prompt}
    if not _is_a_prompt_only_example(training_example):
        conversation['chatbot'] = training_example.response
    return formatter_configs.llama_guard_prompt_configs.instructions_format_string.format_map({'guidelines': full_guidelines_text, 'conversation': _serialize_conversation(conversation)})