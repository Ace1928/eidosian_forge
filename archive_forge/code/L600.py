import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re
import os


class TextSummarizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Summarizer")
        self.root.geometry("800x600")

        model_path = "/home/lloyd/Downloads/local_model_store/t5-small"
        if os.path.exists(model_path):
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        else:
            self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
            self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
            self.tokenizer.save_pretrained(model_path)
            self.model.save_pretrained(model_path)

        self.create_widgets()

    def create_widgets(self):
        self.load_button = tk.Button(
            self.root, text="Load Text File", command=self.load_file
        )
        self.load_button.pack(pady=10)

        self.text_area = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, width=100, height=20
        )
        self.text_area.pack(pady=10)

        self.summarize_button = tk.Button(
            self.root, text="Summarize", command=self.summarize_text
        )
        self.summarize_button.pack(pady=10)

        self.summary_area = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, width=100, height=10
        )
        self.summary_area.pack(pady=10)

        self.save_button = tk.Button(
            self.root, text="Save Summary", command=self.save_summary
        )
        self.save_button.pack(pady=10)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, "r", encoding="utf-8") as file:
                raw_text = file.read()
                cleaned_text = self.clean_text(raw_text)
                self.text_area.delete(1.0, tk.END)
                self.text_area.insert(tk.END, cleaned_text)

    def clean_text(self, text):
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def summarize_text(self):
        article = self.text_area.get(1.0, tk.END).strip()
        if not article:
            messagebox.showwarning("Warning", "No text to summarize")
            return

        chunk_size = 1000
        max_summary_length = 250
        chunks = [
            article[i : i + chunk_size] for i in range(0, len(article), chunk_size)
        ]
        summaries = []

        for chunk in chunks:
            input_text = "summarize: " + chunk
            input_ids = self.tokenizer.encode(
                input_text, return_tensors="pt", max_length=chunk_size, truncation=True
            )
            summary_ids = self.model.generate(
                input_ids,
                num_beams=4,
                max_length=max_summary_length,
                early_stopping=True,
            )
            summary = self.tokenizer.decode(
                summary_ids.squeeze(), skip_special_tokens=True
            )
            summaries.append(summary)
            self.summary_area.insert(tk.END, summary + "\n\n")

        while len(summaries) > 1:
            combined_summary = " ".join(summaries)
            input_text = "summarize: " + combined_summary
            input_ids = self.tokenizer.encode(
                input_text, return_tensors="pt", max_length=chunk_size, truncation=True
            )
            summary_ids = self.model.generate(
                input_ids,
                num_beams=4,
                max_length=max_summary_length,
                early_stopping=True,
            )
            summary = self.tokenizer.decode(
                summary_ids.squeeze(), skip_special_tokens=True
            )
            summaries = [summary]

        final_summary = summaries[0]
        self.summary_area.insert(tk.END, "\nFinal Summary:\n" + final_summary)

    def save_summary(self):
        summary = self.summary_area.get(1.0, tk.END).strip()
        if not summary:
            messagebox.showwarning("Warning", "No summary to save")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("Text files", "*.txt")]
        )
        if file_path:
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(summary)
            messagebox.showinfo("Info", "Summary saved successfully")


if __name__ == "__main__":
    root = tk.Tk()
    app = TextSummarizerApp(root)
    root.mainloop()
