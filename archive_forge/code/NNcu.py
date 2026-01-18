import tkinter as tk
from tkinter import filedialog, messagebox, Label, Button, Frame, Canvas, Entry
import logging
import math
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class PyramidNumbersApp:
    def __init__(self, root: tk.Tk) -> None:
        """
        Initialize the PyramidNumbersApp with the given Tk root.

        Args:
            root (tk.Tk): The root window for the application.
        """
        self.root = root
        self.root.title("Pyramid Numbers Decoder")
        self.setup_ui()

    def setup_ui(self) -> None:
        """
        Set up the user interface for the Pyramid Numbers application.
        """
        self.frame = Frame(self.root)
        self.frame.pack(padx=10, pady=10)

        self.input_text_entry = Entry(self.frame, width=50)
        self.input_text_entry.pack(fill=tk.X)
        self.input_text_entry.insert(0, "Enter text or load a file...")

        self.load_button = Button(self.frame, text="Load File", command=self.load_file)
        self.load_button.pack(fill=tk.X)

        self.decode_button = Button(
            self.frame,
            text="Decode Pyramid Numbers",
            command=self.decode_pyramid_numbers,
            state=tk.NORMAL,
        )
        self.decode_button.pack(fill=tk.X)

        self.message_label = Label(self.frame, text="Decoded message will appear here.")
        self.message_label.pack(fill=tk.X)

        self.canvas = Canvas(self.root, width=600, height=800)
        self.canvas.pack(side=tk.RIGHT)

    def load_file(self) -> None:
        """
        Prompt the user to select a file and load its content into the input text entry.
        """
        logging.info("Prompting user to select a file.")
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, "r") as file:
                content = file.read()
                self.input_text_entry.delete(0, tk.END)
                self.input_text_entry.insert(0, content)
            logging.info(f"File loaded: {file_path}")
        else:
            messagebox.showinfo("No file selected", "Please select a valid text file.")
            logging.warning("No file selected by the user.")

    def decode_pyramid_numbers(self) -> None:
        """
        Decode the pyramid numbers from the input text and display the corresponding message.
        """
        logging.info("Decoding pyramid numbers.")
        input_text = self.input_text_entry.get()
        message = self.decode_pyramid_message(input_text)
        self.display_message(message)

    def decode_pyramid_message(self, input_text: str) -> str:
        """
        Decode the message based on pyramid (triangular) number logic.

        Args:
            input_text (str): The input text containing numbers and corresponding words.

        Returns:
            str: The decoded message.
        """

        def is_triangular(n: int) -> bool:
            x = (-1 + math.sqrt(1 + 8 * n)) / 2
            return x.is_integer()

        lines = input_text.strip().split("\n")
        number_word_map: Dict[int, str] = {}
        max_number = 0

        for line in lines:
            parts = line.split()
            number = int(parts[0])
            word = " ".join(parts[1:])
            number_word_map[number] = word
            if number > max_number:
                max_number = number

        words = [
            number_word_map[i]
            for i in range(1, max_number + 1)
            if is_triangular(i) and i in number_word_map
        ]
        return " ".join(words)

    def display_message(self, message: str) -> None:
        """
        Display the given message in the message label.

        Args:
            message (str): The message to display.
        """
        self.message_label.config(text=f"Decoded message: {message}")
        logging.info(f"Message displayed: {message}")


# Example usage
root = tk.Tk()
app = PyramidNumbersApp(root)
root.mainloop()
