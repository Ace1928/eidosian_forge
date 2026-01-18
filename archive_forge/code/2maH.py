import sys
import logging
from typing import List, Tuple, Generator
from functools import wraps
from time import perf_counter
from contextlib import contextmanager
from memory_profiler import profile
import multiprocessing
import argparse
import itertools
import functools
import psutil

# Import necessary modules for GUI development
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Importing necessary modules for Unicode character handling, color formatting, performance profiling, resource management, parallel processing, and user interaction
from typing import List, Generator


def log_execution_time(func):
    """
    Decorator to log the execution time of a function.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        execution_time = end_time - start_time
        logging.info(f"Execution time of {func.__name__}: {execution_time:.5f} seconds")
        return result

    return wrapper


def log_memory_usage_decorator(func):
    """
    Decorator to log the memory usage before and after executing a function.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        before = psutil.Process().memory_info().rss
        result = func(*args, **kwargs)
        after = psutil.Process().memory_info().rss
        logging.info(
            f"Memory usage of {func.__name__}: {before} bytes -> {after} bytes"
        )
        return result

    return wrapper


# Define the function to programmatically generate characters using memory-efficient generators
@log_execution_time
@log_memory_usage_decorator
def generate_utf_characters(
    custom_ranges: List[Tuple[int, int]] = None
) -> Generator[str, None, None]:
    """
    Generates a comprehensive list of UTF block, pipe, shape, and other related characters by systematically iterating through Unicode code points using memory-efficient generators.

    Args:
        custom_ranges (List[Tuple[int, int]], optional): Custom Unicode ranges to include in the generation process. Defaults to None.

    Returns:
        Generator[str, None, None]: A generator yielding unique UTF characters including block, pipe, shape, and other related characters.
    """
    default_ranges: List[Tuple[int, int]] = [
        (0x2500, 0x257F),  # Box Drawing
        (0x2580, 0x259F),  # Block Elements
        (0x25A0, 0x25FF),  # Geometric Shapes
        (0x2600, 0x26FF),  # Miscellaneous Symbols
        (0x2700, 0x27BF),  # Dingbats
        (0x1F300, 0x1F5FF),  # Miscellaneous Symbols and Pictographs
        (0x1F600, 0x1F64F),  # Emoticons
        (0x1F680, 0x1F6FF),  # Transport and Map Symbols
        (0x1F900, 0x1F9FF),  # Supplemental Symbols and Pictographs
    ]

    # Use custom ranges if provided, otherwise use default ranges
    unicode_ranges = custom_ranges if custom_ranges else default_ranges

    # Iterate through the defined Unicode ranges using memory-efficient generators
    for start, end in unicode_ranges:
        for code_point in range(start, end + 1):
            try:
                character = chr(code_point)  # Convert code point to character
                yield character  # Yield character from the generator
            except ValueError as e:
                logging.error(
                    f"Skipping invalid or non-representable code point: {code_point}. Error: {str(e)}"
                )


def generate_color_gradient(start_hex: str, end_hex: str, steps: int) -> List[str]:
    """
    Generates a gradient list of colors between two hexadecimal colors.

    Args:
        start_hex (str): The starting color in hexadecimal format.
        end_hex (str): The ending color in hexadecimal format.
        steps (int): The number of colors to generate in the gradient.

    Returns:
        List[str]: A list of hexadecimal color codes.
    """
    start_rgb = tuple(int(start_hex[i : i + 2], 16) for i in (1, 3, 5))
    end_rgb = tuple(int(end_hex[i : i + 2], 16) for i in (1, 3, 5))

    gradient = []
    for step in range(steps):
        interpolated = tuple(
            start + (end - start) * step // (steps - 1)
            for start, end in zip(start_rgb, end_rgb)
        )
        gradient.append("#" + "".join(f"{value:02x}" for value in interpolated))

    return gradient


def create_color_spectrum() -> List[str]:
    """
    Creates a comprehensive color spectrum by combining multiple color gradients.

    Returns:
        List[str]: A list of hexadecimal color codes covering the entire spectrum.
    """
    spectrum = []
    # Define key colors in the spectrum
    colors = [
        ("#000000", "#808080"),  # Black to gray
        ("#808080", "#FFFFFF"),  # Gray to white
        ("#FF0000", "#FFA500"),  # Red to orange
        ("#FFA500", "#FFFF00"),  # Orange to yellow
        ("#008000", "#0000FF"),  # Green to blue
        ("#4B0082", "#EE82EE"),  # Indigo to violet
        ("#EE82EE", "#FF00FF"),  # Violet to fuchsia
    ]
    steps_per_gradient = 20

    for start_color, end_color in colors:
        spectrum.extend(
            generate_color_gradient(start_color, end_color, steps_per_gradient)
        )

    return spectrum


# Define a context manager to log memory usage
@contextmanager
def log_memory_usage():
    """
    Context manager to log memory usage before and after a code block.
    """
    before = psutil.Process().memory_info().rss
    try:
        yield
    finally:
        after = psutil.Process().memory_info().rss
        logging.info(f"Memory usage: {before} bytes -> {after} bytes")


# Define a class to handle character printing with colors
class CharacterPrinter:
    """
    A class to handle printing characters with colors.
    """

    def __init__(self, characters: Generator[str, None, None]):
        """
        Initialize the CharacterPrinter.

        Args:
            characters (Generator[str, None, None]): A generator yielding characters to print.
        """
        self.characters = characters
        self.colors = create_color_spectrum()

    def print_characters(self, batch_size: int = 1000):
        """
        Print the characters with colors in batches.

        Args:
            batch_size (int, optional): The number of characters to print in each batch. Defaults to 1000.
        """
        with log_memory_usage():
            for batch in self._batch(self.characters, batch_size):
                for char, color in zip(batch, itertools.cycle(self.colors)):
                    self.character_text.insert("end", f"{char}", ("color", color))
                    try:
                        self.character_text.tag_configure("color", foreground=color)
                    except tk.TclError as e:
                        logging.error(f"Invalid color '{color}': {e}")
                self.character_text.insert("end", "\n")

    @staticmethod
    def _batch(iterable, batch_size):
        """
        Divide an iterable into batches of a specified size.

        Args:
            iterable (Iterable): The iterable to divide into batches.
            batch_size (int): The size of each batch.

        Yields:
            List: A batch of elements from the iterable.
        """
        iterator = iter(iterable)
        while True:
            batch = list(itertools.islice(iterator, batch_size))
            if not batch:
                break
            yield batch


# Define the main GUI class
class UnicodeCharacterGeneratorGUI(tk.Tk):
    """
    A class representing the main GUI window for the Unicode character generator.
    """

    def __init__(self):
        """
        Initialize the main GUI window.
        """
        super().__init__()

        # Set window title and size
        self.title("Unicode Character Generator")
        self.geometry("800x600")

        # Create and configure widgets
        self.create_widgets()

    def create_widgets(self):
        """
        Create and configure the GUI widgets.
        """
        # Create a frame for the input fields
        input_frame = ttk.Frame(self)
        input_frame.pack(pady=10)

        # Create a label and entry for custom Unicode ranges
        ttk.Label(input_frame, text="Custom Unicode Ranges:").grid(
            row=0, column=0, sticky="e"
        )
        self.ranges_entry = ttk.Entry(input_frame, width=50)
        self.ranges_entry.grid(row=0, column=1, padx=5)

        # Create a label and dropdown for output destination
        ttk.Label(input_frame, text="Output Destination:").grid(
            row=1, column=0, sticky="e"
        )
        self.output_var = tk.StringVar(value="console")
        ttk.OptionMenu(input_frame, self.output_var, "console", "console", "file").grid(
            row=1, column=1, padx=5, sticky="w"
        )

        # Create a label and entry for output file name
        ttk.Label(input_frame, text="Output File Name:").grid(
            row=2, column=0, sticky="e"
        )
        self.file_entry = ttk.Entry(input_frame, width=50)
        self.file_entry.insert(0, "characters.txt")
        self.file_entry.grid(row=2, column=1, padx=5)

        # Create a label and entry for batch size
        ttk.Label(input_frame, text="Batch Size:").grid(row=3, column=0, sticky="e")
        self.batch_size_entry = ttk.Entry(input_frame, width=10)
        self.batch_size_entry.insert(0, "1000")
        self.batch_size_entry.grid(row=3, column=1, padx=5, sticky="w")

        # Create a label and dropdown for character size
        ttk.Label(input_frame, text="Character Size:").grid(row=4, column=0, sticky="e")
        self.size_var = tk.StringVar(value="12")
        ttk.OptionMenu(
            input_frame,
            self.size_var,
            "12",
            "8",
            "10",
            "12",
            "14",
            "16",
            "18",
            "20",
            "24",
            "28",
            "32",
        ).grid(row=4, column=1, padx=5, sticky="w")

        # Create a label and dropdown for font type
        ttk.Label(input_frame, text="Font Type:").grid(row=5, column=0, sticky="e")
        self.font_var = tk.StringVar(value="Arial")
        ttk.OptionMenu(
            input_frame,
            self.font_var,
            "Arial",
            "Arial",
            "Courier",
            "Helvetica",
            "Times",
            "Verdana",
        ).grid(row=5, column=1, padx=5, sticky="w")

        # Create a button to generate and print characters
        ttk.Button(
            self, text="Generate and Print", command=self.generate_and_print
        ).pack(pady=10)

        # Create a text widget to display the generated characters
        self.character_text = tk.Text(self, wrap="word", width=80, height=20)
        self.character_text.pack(padx=10, pady=10)

        # Create a scrollbar for the text widget
        scrollbar = ttk.Scrollbar(self, command=self.character_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.character_text.configure(yscrollcommand=scrollbar.set)

    def generate_and_print(self):
        """
        Generate and print Unicode characters based on user input.
        """
        # Get user input values
        custom_ranges_str = self.ranges_entry.get()
        output_destination = self.output_var.get()
        output_file = self.file_entry.get()
        batch_size = int(self.batch_size_entry.get())
        character_size = int(self.size_var.get())
        font_type = self.font_var.get()

        # Set the font for the text widget
        self.character_text.configure(font=(font_type, character_size))

        # Process custom Unicode ranges
        custom_ranges = []
        for range_str in custom_ranges_str.split(","):
            range_str = range_str.strip()
            if range_str:
                start, end = map(lambda x: int(x, 16), range_str.split("-"))
                custom_ranges.append((start, end))

        # Generate UTF characters based on the provided ranges
        characters = generate_utf_characters(custom_ranges)

        # Generate the color spectrum
        colors = self.generate_color_spectrum()

        # Print or save the characters based on the output option
        if output_destination == "console":
            # Clear the text widget
            self.character_text.delete("1.0", "end")

            # Use parallel processing to speed up character printing
            with multiprocessing.Pool() as pool:
                batches = itertools.islice(characters, batch_size)
                for batch in batches:
                    # Print characters in the text widget
                    for char, color in zip(batch, itertools.cycle(colors)):
                        self.character_text.insert("end", f"{char}", ("color", color))
                        try:
                            self.character_text.tag_configure("color", foreground=color)
                        except tk.TclError as e:
                            logging.error(f"Invalid color '{color}': {e}")
                            # Optionally, set to a default color if needed
                            # self.character_text.tag_configure("color", foreground="black")
                    self.character_text.insert("end", "\n")
        else:
            # Save characters to a file
            with open(output_file, "w", encoding="utf-8") as file:
                for char in characters:
                    file.write(f"{char}\n")
            messagebox.showinfo("File Saved", f"Characters saved to {output_file}")

    def generate_color_spectrum(self, steps: int = 256) -> List[str]:
        """
        Generates a color spectrum.

        Args:
            steps (int): The number of colors to generate.

        Returns:
            List[str]: A list of hexadecimal color codes.
        """
        spectrum = []
        # Define key colors in the spectrum
        colors = [
            "#000000",
            "#808080",  # Black to gray
            "#FFFFFF",  # White
            "#FF0000",
            "#FFA500",  # Red to orange
            "#FFFF00",  # Yellow
            "#008000",
            "#0000FF",  # Green to blue
            "#4B0082",
            "#EE82EE",  # Indigo to violet
            "#FF00FF",  # Fuchsia
        ]
        color_transition_steps = steps // (len(colors) - 1)

        for i in range(len(colors) - 1):
            start_color = colors[i]
            end_color = colors[i + 1]
            for step in range(color_transition_steps):
                # Calculate the intermediate color
                start_rgb = tuple(int(start_color[j : j + 2], 16) for j in (1, 3, 5))
                end_rgb = tuple(int(end_color[j : j + 2], 16) for j in (1, 3, 5))
                interp_rgb = tuple(
                    start + (end - start) * step // (color_transition_steps - 1)
                    for start, end in zip(start_rgb, end_rgb)
                )
                spectrum.append("#" + "".join(f"{value:02x}" for value in interp_rgb))

        return spectrum

    def run(self):
        """
        Run the main event loop of the GUI.
        """
        self.mainloop()


# Run the main GUI window
if __name__ == "__main__":
    gui = UnicodeCharacterGeneratorGUI()
    gui.run()
