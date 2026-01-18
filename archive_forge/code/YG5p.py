import os
import hashlib
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from typing import List, Dict, Tuple
from collections import defaultdict
import openai
from openai import OpenAI
from typing_extensions import override
from openai import AssistantEventHandler
import textwrap
import logging

logging.basicConfig(
    filename="filechecker.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class FileChecker:
    def __init__(self, master):
        self.master = master
        master.title("Intelligent File Similarity Checker")

        self.folder_path = tk.StringVar()
        self.similarity_threshold = tk.DoubleVar(value=0.8)
        self.output_folder_path = tk.StringVar()

        self.create_widgets()

        # Initialize OpenAI client
        self.openai_client = OpenAI()

        # Create an assistant for generating file names
        self.naming_assistant = self.openai_client.beta.assistants.create(
            name="File Naming Assistant",
            instructions="You are an assistant that generates concise, accurate, and specific file names based on the contents of a document. The file names should reflect the main topic or purpose of the document and be succinct and unique.",
            model="gpt-4-turbo",
        )

    def create_widgets(self):
        # Folder selection
        folder_frame = ttk.Frame(self.master)
        folder_frame.pack(padx=10, pady=10)

        ttk.Label(folder_frame, text="Select Input Folder:").grid(
            row=0, column=0, padx=5, pady=5, sticky=tk.W
        )
        ttk.Entry(folder_frame, textvariable=self.folder_path, width=50).grid(
            row=0, column=1, padx=5, pady=5
        )
        ttk.Button(folder_frame, text="Browse", command=self.browse_input_folder).grid(
            row=0, column=2, padx=5, pady=5
        )

        # Output folder selection
        output_folder_frame = ttk.Frame(self.master)
        output_folder_frame.pack(padx=10, pady=10)

        ttk.Label(output_folder_frame, text="Select Output Folder:").grid(
            row=0, column=0, padx=5, pady=5, sticky=tk.W
        )
        ttk.Entry(
            output_folder_frame, textvariable=self.output_folder_path, width=50
        ).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(
            output_folder_frame, text="Browse", command=self.browse_output_folder
        ).grid(row=0, column=2, padx=5, pady=5)

        # Similarity threshold
        threshold_frame = ttk.Frame(self.master)
        threshold_frame.pack(padx=10, pady=10)

        ttk.Label(threshold_frame, text="Similarity Threshold:").grid(
            row=0, column=0, padx=5, pady=5, sticky=tk.W
        )
        ttk.Scale(
            threshold_frame,
            from_=0.0,
            to=1.0,
            variable=self.similarity_threshold,
            resolution=0.01,
            orient=tk.HORIZONTAL,
        ).grid(row=0, column=1, padx=5, pady=5)

        # Action buttons
        button_frame = ttk.Frame(self.master)
        button_frame.pack(padx=10, pady=10)

        ttk.Button(
            button_frame, text="Check Similarity", command=self.check_similarity
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Quit", command=self.master.quit).pack(
            side=tk.LEFT, padx=5
        )

    def browse_input_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.folder_path.set(folder_selected)

    def browse_output_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.output_folder_path.set(folder_selected)

    def check_similarity(self):
        input_folder = self.folder_path.get()
        output_folder = self.output_folder_path.get()
        if not input_folder:
            messagebox.showerror("Error", "Please select an input folder.")
            return
        if not output_folder:
            messagebox.showerror("Error", "Please select an output folder.")
            return

        threshold = self.similarity_threshold.get()

        file_hashes = self.calculate_file_hashes(input_folder)
        similarity_groups = self.group_similar_files(file_hashes, threshold)

        self.display_results(similarity_groups, output_folder)

    def calculate_file_hashes(self, folder: str) -> Dict[str, str]:
        file_hashes = {}
        for root, dirs, files in os.walk(folder):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                    file_hashes[file_path] = file_hash
        return file_hashes

    def group_similar_files(
        self, file_hashes: Dict[str, str], threshold: float
    ) -> Dict[str, List[str]]:
        similarity_groups = defaultdict(list)
        for file_path, file_hash in file_hashes.items():
            for group_hash, group_files in similarity_groups.items():
                if self.compare_hashes(file_hash, group_hash) >= threshold:
                    group_files.append(file_path)
                    break
            else:
                similarity_groups[file_hash].append(file_path)
        return similarity_groups

    def compare_hashes(self, hash1: str, hash2: str) -> float:
        if len(hash1) != len(hash2):
            return 0.0

        matching_chars = sum(1 for c1, c2 in zip(hash1, hash2) if c1 == c2)
        return matching_chars / len(hash1)

    def display_results(
        self, similarity_groups: Dict[str, List[str]], output_folder: str
    ):
        result_window = tk.Toplevel(self.master)
        result_window.title("Similarity Results")

        tree = ttk.Treeview(result_window, columns=("Files",))
        tree.heading("Files", text="Similar Files")
        tree.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        for group_hash, group_files in similarity_groups.items():
            if len(group_files) > 1:
                group_node = tree.insert("", tk.END, text=f"Group {group_hash[:8]}")
                for file_path in group_files:
                    tree.insert(group_node, tk.END, text=file_path)

                    # Generate file name using OpenAI assistant
                    with open(file_path, "r") as file:
                        file_content = file.read()
                        generated_name = self.generate_file_name(file_content)
                        new_file_path = os.path.join(output_folder, generated_name)
                        os.rename(file_path, new_file_path)
                        tree.insert(
                            group_node, tk.END, text=f"Renamed to: {generated_name}"
                        )

        if len(tree.get_children()) == 0:
            tree.insert("", tk.END, text="No similar files found.")

    def generate_file_name(self, file_content: str) -> str:
        # Create a thread for the conversation with the naming assistant
        thread = self.openai_client.beta.threads.create()

        # Split file content into chunks of 4096 characters or less
        content_chunks = textwrap.wrap(file_content, width=4096)

        for chunk in content_chunks:
            # Add each chunk as a message to the thread
            message = self.openai_client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=f"Please generate a concise, accurate, and specific file name for the following document chunk:\n\n{chunk}",
            )

        # Create and stream the run to get the generated file name
        class EventHandler(AssistantEventHandler):
            def __init__(self):
                self.generated_name = ""

            @override
            def on_text_delta(self, delta, snapshot):
                self.generated_name += delta.value

        event_handler = EventHandler()

        with self.openai_client.beta.threads.runs.stream(
            thread_id=thread.id,
            assistant_id=self.naming_assistant.id,
            event_handler=event_handler,
        ) as stream:
            stream.until_done()

        generated_name = event_handler.generated_name.strip()
        logging.info(f"Generated file name: {generated_name}")
        return generated_name


root = tk.Tk()
app = FileChecker(root)
root.mainloop()
