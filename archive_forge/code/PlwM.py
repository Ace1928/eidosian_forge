import os
from datasets import load_dataset
from convokit import Corpus, TextParser, PolitenessStrategies, download
import pandas as pd

# Define directories
RAW_DATA_DIR = "/home/lloyd/Downloads/indegodata/raw_datasets"
PROCESSED_DATA_DIR = "/home/lloyd/Downloads/indegodata/processed_data"

# Datasets from Huggingface for training and testing models
HUGGINGFACE_URLS = [
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

# List of datasets to download from ConvoKit
CONVOKIT_DATASETS = [
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


# Function to download and process Huggingface datasets
def process_huggingface_datasets(urls, raw_data_dir, processed_data_dir):
    corpora = {}
    for url in urls:
        dataset_name = url.split("/")[-1]
        dataset = load_dataset(dataset_name)
        dataset.save_to_disk(os.path.join(raw_data_dir, dataset_name))

        # Convert to ConvoKit corpus
        convokit_corpus = Corpus.from_dataset(dataset)
        convokit_corpus.dump(os.path.join(processed_data_dir, dataset_name))
        corpora[dataset_name] = convokit_corpus

    # Merging datasets into a universal corpus
    universal_corpus = corpora[urls[0].split("/")[-1]]
    for url in urls[1:]:
        dataset_name = url.split("/")[-1]
        universal_corpus = universal_corpus.merge(corpora[dataset_name])

    # Save the merged corpus
    universal_corpus.dump(os.path.join(processed_data_dir, "huggingface_corpus"))

    return universal_corpus


# Function to download and process ConvoKit datasets
def process_convokit_datasets(datasets, raw_data_dir, processed_data_dir):
    corpora = {}
    for dataset in datasets:
        corpora[dataset] = Corpus(filename=download(dataset, data_dir=raw_data_dir))

    # Merging datasets into a universal corpus
    universal_corpus = corpora[datasets[0]]
    for dataset in datasets[1:]:
        universal_corpus = universal_corpus.merge(corpora[dataset])

    # Save the merged corpus
    universal_corpus.dump(os.path.join(processed_data_dir, "universal_corpus"))

    return universal_corpus


# Process Huggingface datasets
huggingface_corpus = process_huggingface_datasets(
    HUGGINGFACE_URLS, RAW_DATA_DIR, PROCESSED_DATA_DIR
)

# Process ConvoKit datasets
convokit_corpus = process_convokit_datasets(
    CONVOKIT_DATASETS, RAW_DATA_DIR, PROCESSED_DATA_DIR
)

# Apply transformers to Huggingface corpus
parser = TextParser()
huggingface_corpus = parser.transform(huggingface_corpus)

politeness = PolitenessStrategies()
huggingface_corpus = politeness.transform(huggingface_corpus)

# Apply transformers to ConvoKit corpus
convokit_corpus = parser.transform(convokit_corpus)
convokit_corpus = politeness.transform(convokit_corpus)

# Converting Huggingface corpus components to DataFrames
huggingface_utterances_df = huggingface_corpus.get_utterances_dataframe()
huggingface_conversations_df = huggingface_corpus.get_conversations_dataframe()
huggingface_speakers_df = huggingface_corpus.get_speakers_dataframe()

# Converting ConvoKit corpus components to DataFrames
convokit_utterances_df = convokit_corpus.get_utterances_dataframe()
convokit_conversations_df = convokit_corpus.get_conversations_dataframe()
convokit_speakers_df = convokit_corpus.get_speakers_dataframe()

# Display the first few rows of the utterances DataFrame
print(huggingface_utterances_df.head())
print(convokit_utterances_df.head())

# Save the dataframes to CSV for further use
huggingface_utterances_df.to_csv(
    os.path.join(PROCESSED_DATA_DIR, "huggingface_utterances.csv"), index=False
)
huggingface_conversations_df.to_csv(
    os.path.join(PROCESSED_DATA_DIR, "huggingface_conversations.csv"), index=False
)
huggingface_speakers_df.to_csv(
    os.path.join(PROCESSED_DATA_DIR, "huggingface_speakers.csv"), index=False
)

convokit_utterances_df.to_csv(
    os.path.join(PROCESSED_DATA_DIR, "convokit_utterances.csv"), index=False
)
convokit_conversations_df.to_csv(
    os.path.join(PROCESSED_DATA_DIR, "convokit_conversations.csv"), index=False
)
convokit_speakers_df.to_csv(
    os.path.join(PROCESSED_DATA_DIR, "convokit_speakers.csv"), index=False
)

# Additional feature extraction and analysis can be added here
