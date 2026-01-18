from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Deci/DeciLM-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.save_pretrained("/home/lloyd/Downloads/local_model_store/DeciLM-7B")
model.save_pretrained("/home/lloyd/Downloads/local_model_store/DeciLM-7B")
