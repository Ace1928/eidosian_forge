def pyramid_numbers(self) -> None:
    """
    This function is used to output the "pyramid" numbers (1, 3, 6, ...).
    It then reads, after prompting the user for a file or input text through a GUI. Within the input text, it then creates a dictionary mapping each of the pyramid numbers to its associated word (according to the file, each line of the file starting with a number and then followed by a word. The number corresponding to the word.)
    The function then outputs the string, in order, of the words corresponding to the pyramid numbers.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox

    def load_file() -> str:
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if not file_path:
            messagebox.showinfo("No file selected", "Please select a valid text file.")
            return ""
        return file_path

    def read_file(file_path: str) -> dict:
        number_word_map = {}
        try:
            with open(file_path, "r") as file:
                for line in file:
                    parts = line.strip().split()
                    if parts:
                        number = int(parts[0])
                        word = " ".join(parts[1:])
                        number_word_map[number] = word
        except Exception as e:
            messagebox.showerror(
                "File Error", f"An error occurred while reading the file: {str(e)}"
            )
        return number_word_map

    def calculate_pyramid_numbers(max_number: int) -> list:
        pyramid_numbers = []
        current_number = 1
        level = 1
        while current_number <= max_number:
            last_number = current_number + level - 1
            pyramid_numbers.append(last_number)
            current_number = last_number + 1
            level += 1
        return pyramid_numbers

    def display_message(words: list) -> None:
        message = " ".join(words)
        messagebox.showinfo("Pyramid Numbers Message", message)

    file_path = load_file()
    if file_path:
        number_word_map = read_file(file_path)
        if number_word_map:
            max_number = max(number_word_map.keys())
            pyramid_nums = calculate_pyramid_numbers(max_number)
            words = [
                number_word_map[num] for num in pyramid_nums if num in number_word_map
            ]
            display_message(words)
