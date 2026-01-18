from dataclasses import dataclass
@dataclass
class alpaca_dataset:
    dataset: str = 'alpaca_dataset'
    train_split: str = 'train'
    test_split: str = 'val'
    data_path: str = 'src/llama_recipes/datasets/alpaca_data.json'