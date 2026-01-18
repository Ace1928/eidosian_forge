from dataclasses import dataclass
@dataclass
class custom_dataset:
    dataset: str = 'custom_dataset'
    file: str = 'examples/custom_dataset.py'
    train_split: str = 'train'
    test_split: str = 'validation'