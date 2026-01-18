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
        This function decodes a message encoded based on pyramid (triangular) number logic.
        The input text is expected to contain lines, each with a number followed by a word or phrase.
        The function maps each number to its corresponding word if the number is a triangular number.
        Triangular numbers are those which can form an equilateral triangle (e.g., 1, 3, 6, 10, ...).
        The function then constructs a message from words mapped from triangular numbers in ascending order.

        Args:
            input_text (str): The input text containing numbers and corresponding words.

        Returns:
            str: The decoded message constructed from words that correspond to triangular numbers.
        """

        # Nested function to check if a number is triangular
        def is_triangular(n: int) -> bool:
            """
            Determines if a number is a triangular number.
            Triangular numbers are calculated by the formula: n(n+1)/2 = x
            Solving for n using the quadratic formula gives us: n = (-1 + sqrt(1 + 8*x)) / 2
            If n is an integer, x is a triangular number.

            Args:
                n (int): The number to check.

            Returns:
                bool: True if n is a triangular number, False otherwise.
            """
            x = (
                -1 + math.sqrt(1 + 8 * n)
            ) / 2  # Calculate the potential triangular number index
            return x.is_integer()  # Check if the index is an integer

        # Process the input text into lines
        lines = input_text.strip().split(
            "\n"
        )  # Strip leading/trailing whitespace and split by new lines
        number_word_map: Dict[int, str] = (
            {}
        )  # Initialize a dictionary to map numbers to words
        max_number = 0  # Variable to keep track of the highest number encountered

        # Iterate over each line to populate the number_word_map
        for line in lines:
            parts = (
                line.split()
            )  # Split the line into parts; expects the first part to be a number
            number = int(parts[0])  # Convert the first part to an integer (the number)
            word = " ".join(
                parts[1:]
            )  # Join the remaining parts as the corresponding word or phrase
            number_word_map[number] = word  # Map the number to the word
            if number > max_number:
                max_number = (
                    number  # Update the maximum number if the current number is larger
                )

        # Construct the decoded message from words corresponding to triangular numbers
        words = [
            number_word_map[i]  # Access the word mapped to the number i
            for i in range(1, max_number + 1)  # Iterate from 1 to the maximum number
            if is_triangular(i)
            and i
            in number_word_map  # Include the word if i is triangular and in the map
        ]
        return " ".join(
            words
        )  # Join the selected words into a single string to form the decoded message

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
