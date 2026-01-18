import tkinter as tk
from tkinter import filedialog, ttk
import zstandard as zstd
from PIL import Image
from pathlib import Path


class ImageCompressorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Compressor & Decompressor")
        self.root.geometry("400x200")

        # Setup UI components
        self.setup_ui()

    def setup_ui(self):
        # File selection button
        self.file_path_var = tk.StringVar()
        file_select_btn = tk.Button(
            self.root, text="Select Image", command=self.select_file
        )
        file_select_btn.pack(pady=10)

        # Compression format dropdown
        self.format_var = tk.StringVar()
        formats = ["zstd"]  # Add more formats if available
        format_dropdown = ttk.Combobox(
            self.root, textvariable=self.format_var, values=formats
        )
        format_dropdown.pack(pady=10)
        format_dropdown.current(0)

        # Compress button
        compress_btn = tk.Button(self.root, text="Compress", command=self.compress)
        compress_btn.pack(pady=10)

        # Decompress button
        decompress_btn = tk.Button(
            self.root, text="Decompress", command=self.decompress
        )
        decompress_btn.pack(pady=10)

    def select_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_path_var.set(file_path)

    def compress(self):
        input_path = Path(self.file_path_var.get())
        if not input_path.exists():
            print("Error: The file does not exist.")
            return

        output_path = input_path.with_suffix(f".{input_path.suffix[1:]}.zstd")
        # Open the image to ensure it's a valid image file
        with Image.open(input_path) as img:
            img_format = img.format.lower()

        # Read the original image data
        with open(input_path, "rb") as file:
            data = file.read()

        # Compress the data
        cctx = zstd.ZstdCompressor()
        compressed_data = cctx.compress(data)

        # Save the compressed data with a .format.zstd extension
        with open(output_path.with_suffix(f".{img_format}.zstd"), "wb") as file:
            file.write(compressed_data)

        print(
            f"Compressed file saved to: {output_path.with_suffix(f'.{img_format}.zstd')}"
        )

    def decompress(self):
        input_path = Path(self.file_path_var.get())
        if not input_path.exists() or not input_path.suffix.endswith("zstd"):
            print("Error: The file does not exist or is not a .zstd file.")
            return

        dctx = zstd.ZstdDecompressor()
        with open(input_path, "rb") as compressed:
            decompressed_data = dctx.decompress(compressed.read())

        output_path = input_path.with_suffix("")  # Remove .zstd extension
        with open(output_path, "wb") as file:
            file.write(decompressed_data)

        print(f"Decompressed file saved to: {output_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCompressorApp(root)
    root.mainloop()
