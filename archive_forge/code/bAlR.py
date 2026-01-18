from transformers import T5Tokenizer, T5ForConditionalGeneration

# News article - This should be selected by the user as a .txt file, which is in the form of raw html scrape data, and then converted to plain text.
article = ""

# Load the T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer.save_pretrained("/home/lloyd/Downloads/local_model_store/t5-small")
model.save_pretrained("/home/lloyd/Downloads/local_model_store/t5-small")


# Prepare the input
input_text = "summarize: " + article
input_ids = tokenizer.encode(
    input_text, return_tensors="pt", max_length=512, truncation=True
)

# Generate the summary
summary_ids = model.generate(
    input_ids, num_beams=4, max_length=100, early_stopping=True
)
# Convert tensor to list and ensure it's 1D
summary_ids = summary_ids.squeeze().tolist()
summary = tokenizer.decode(summary_ids, skip_special_tokens=True)

# Print the summary
print("Summary:", summary)
