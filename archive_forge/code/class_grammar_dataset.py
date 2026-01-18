from dataclasses import dataclass
@dataclass
class grammar_dataset:
    dataset: str = 'grammar_dataset'
    train_split: str = 'src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv'
    test_split: str = 'src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv'