import os
from datasets import load_dataset
from convokit import Corpus, download

# Define directories
raw_data_dir = "/home/lloyd/Downloads/indegodata/raw_datasets"
processed_data_dir = "/home/lloyd/Downloads/indegodata/processed_data"

# Datasets from Huggingface for training and testing models
URLS = [
    "https://huggingface.co/datasets/ambig_qa",
    "https://huggingface.co/datasets/break_data",
    "https://huggingface.co/datasets/tau/commonsense_qa",
    "https://huggingface.co/datasets/stanfordnlp/coqa",
    "https://huggingface.co/datasets/ucinlp/drop",
    "https://huggingface.co/datasets/HongzheBi/DuReader2.0",
    "https://huggingface.co/datasets/hotpot_qa",
    "https://huggingface.co/datasets/narrativeqa",
    "https://huggingface.co/datasets/natural_questions",
    "https://huggingface.co/datasets/newsqa",
    "https://huggingface.co/datasets/allenai/openbookqa",
    "https://huggingface.co/datasets/allenai/qasc",
    "https://huggingface.co/datasets/quac",
    "https://huggingface.co/datasets/rajpurkar/squad_v2",
    "https://huggingface.co/datasets/trec",
    "https://huggingface.co/datasets/tydiqa",
    "https://huggingface.co/datasets/wiki_qa",
    "https://huggingface.co/datasets/ubuntu_dialogs_corpus",
    "https://huggingface.co/datasets/conv_ai",
]

# Download and process datasets
corpora = {}
for url in URLS:
    dataset_name = url.split("/")[-1]
    dataset = load_dataset(dataset_name)
    dataset.save_to_disk(os.path.join(raw_data_dir, dataset_name))

    # Convert to ConvoKit corpus
    convokit_corpus = Corpus.from_dataset(dataset)
    convokit_corpus.dump(os.path.join(processed_data_dir, dataset_name))

# Merging datasets into a universal corpus
huggingface_corpus = Corpus(
    filename=os.path.join(processed_data_dir, URLS[0].split("/")[-1])
)
for url in URLS[1:]:
    dataset_name = url.split("/")[-1]
    next_corpus = Corpus(filename=os.path.join(processed_data_dir, dataset_name))
    huggingface_corpus = huggingface_corpus.merge(next_corpus)

# Save the merged corpus
huggingface_corpus.dump(os.path.join(processed_data_dir, "huggingface_corpus"))

# Import necessary libraries
from convokit import Corpus, download, TextParser, PolitenessStrategies
import pandas as pd

# Define the download directory
raw_data_dir = "/home/lloyd/Downloads/indegodata/raw_datasets"
processed_data_dir = "/home/lloyd/Downloads/indegodata/processed_data"

# List of datasets to download
datasets = [
    "supreme-corpus",
    "wiki-corpus",
    "reddit-corpus-small",
    "chromium-corpus",
    "winning-args-corpus",
    "reddit-coarse-discourse-corpus",
    "persuasionforgood-corpus",
    "iq2-corpus",
    "friends-corpus",
    "switchboard-corpus",
    "wikipedia-politeness-corpus",
    "stack-exchange-politeness-corpus",
    "diplomacy-corpus",
    "gap-corpus",
    "casino-corpus",
]

# Download and save datasets
corpora = {}
for dataset in datasets:
    corpora[dataset] = Corpus(filename=download(dataset, data_dir=raw_data_dir))

# Merging datasets into a universal corpus
universal_corpus = corpora["supreme-corpus"]
for dataset in datasets[1:]:
    universal_corpus = universal_corpus.merge(corpora[dataset])

# Apply transformers
parser = TextParser()
universal_corpus = parser.transform(universal_corpus)

politeness = PolitenessStrategies()
universal_corpus = politeness.transform(universal_corpus)

# Converting corpus components to DataFrames
utterances_df = universal_corpus.get_utterances_dataframe()
conversations_df = universal_corpus.get_conversations_dataframe()
speakers_df = universal_corpus.get_speakers_dataframe()

# Display the first few rows of the utterances DataFrame
print(utterances_df.head())

# Save the dataframes to CSV for further use
utterances_df.to_csv(f"{processed_data_dir}/universal_utterances.csv", index=False)
conversations_df.to_csv(
    f"{processed_data_dir}/universal_conversations.csv", index=False
)
speakers_df.to_csv(f"{processed_data_dir}/universal_speakers.csv", index=False)

# Additional feature extraction and analysis can be added here
