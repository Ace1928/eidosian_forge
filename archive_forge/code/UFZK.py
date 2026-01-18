from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "sshleifer/distilbart-cnn-12-6"
model_path = "/home/lloyd/Downloads/local_model_store/distilbart-cnn-12-6"

# Load and save the model and tokenizer to a directory
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer.save_pretrained(model_path)
model.save_pretrained(model_path)
TEXT_LENGTH = 1000
MAX_LENGTH = TEXT_LENGTH // 2
MIN_LENGTH = TEXT_LENGTH // 4

# Example text
text = """
To use the `sshleifer/distilbart-cnn-12-6` model for text summarization and ensure that it loads from a local model store each time by default (minimizing or eliminating network requirements), you will need to follow these steps:

1. **Download the Model and Tokenizer**: First, download the model and tokenizer to your local machine. This step requires an initial network connection to download the files, but subsequent uses will not.

2. **Set Up the Code**: Modify your Python script to use the local copies of the model and tokenizer.

Here's how you can do it:

### Step 1: Download the Model and Tokenizer

You can download the model and tokenizer using the following commands in a Python script or directly in your Python environment:

### Step 2: Modify the Python Script

Now, modify your script to load the model and tokenizer from the local store by default. Here's how you can adjust your `textsummarygensim.py` script:

This script ensures that the model and tokenizer are loaded from the local directory specified by `model_path`, and the `local_files_only=True` parameter ensures that the Transformers library does not attempt to download files from the internet.

By following these steps, you can use the `sshleifer/distilbart-cnn-12-6` model for summarization while ensuring that it consistently loads from a local store, minimizing network requirements after the initial setup.
"""


def summarize_text(text, model_path=model_path):
    # Load the model and tokenizer from the local model store
    TEXT_LENGTH = len(text)
    MAX_LENGTH = TEXT_LENGTH // 2
    MIN_LENGTH = TEXT_LENGTH // 4
    local_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    local_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path, local_files_only=True
    )

    # Create the summarization pipeline using the local model and tokenizer
    summarizer = pipeline("summarization", model=local_model, tokenizer=local_tokenizer)

    # Perform summarization
    summary = summarizer(
        text,
        max_length=MAX_LENGTH,
        min_length=MIN_LENGTH,
    )

    # Extract and return the summary text
    return summary[0]["summary_text"]


print("Summary:", summarize_text(text))
