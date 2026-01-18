# database.py
import aiosqlite
import asyncio
from cryptography.fernet import Fernet
import os

# Load the encryption key from an environment variable or file
encryption_key = os.getenv("ENCRYPTION_KEY")
cipher_suite = Fernet(encryption_key)


async def init_db():
    async with aiosqlite.connect(os.getenv("DB_PATH", "image_db.sqlite")) as db:
        await db.execute(
            """CREATE TABLE IF NOT EXISTS images
               (hash TEXT PRIMARY KEY, compressed_data BLOB)"""
        )
        await db.commit()


async def insert_compressed_image(hash: str, data: bytes):
    encrypted_data = cipher_suite.encrypt(data)
    async with aiosqlite.connect(os.getenv("DB_PATH", "image_db.sqlite")) as db:
        await db.execute(
            "INSERT OR REPLACE INTO images (hash, compressed_data) VALUES (?, ?)",
            (hash, encrypted_data),
        )
        await db.commit()


async def get_compressed_image(hash: str) -> bytes:
    async with aiosqlite.connect(os.getenv("DB_PATH", "image_db.sqlite")) as db:
        async with db.execute(
            "SELECT compressed_data FROM images WHERE hash = ?", (hash,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                encrypted_data = row[0]
                return cipher_suite.decrypt(encrypted_data)
            return None


# image_processing.py
from PIL import Image
import io
import lzma
import hashlib


def resize_image(image, max_size=(800, 600)):
    image.thumbnail(max_size, Image.ANTIALIAS)
    return image


def compress_image_data(image_data):
    return lzma.compress(image_data)


def decompress_image_data(compressed_data):
    return lzma.decompress(compressed_data)


def get_image_hash(image_data):
    return hashlib.sha256(image_data).hexdigest()


# gui.py
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import io
import asyncio
from threading import Thread
from image_processing import (
    resize_image,
    compress_image_data,
    decompress_image_data,
    get_image_hash,
)
from database import init_db, insert_compressed_image, get_compressed_image


class ImageCompressorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Compressor")
        self.init_widgets()
        self.original_image_data = None
        self.img_hash = None
        # Initialize the database asynchronously
        Thread(target=self.async_init_db, daemon=True).start()

    def async_init_db(self):
        asyncio.run(init_db())

    def init_widgets(self):
        self.load_btn = tk.Button(self, text="Load Image", command=self.load_image)
        self.load_btn.pack()

        self.compress_btn = tk.Button(
            self,
            text="Compress & Store Image",
            state="disabled",
            command=self.compress_image,
        )
        self.compress_btn.pack()

        self.decompress_btn = tk.Button(
            self, text="Decompress & Show Image", command=self.decompress_and_show
        )
        self.decompress_btn.pack()

        self.original_img_label = tk.Label(self)
        self.original_img_label.pack(side="left", padx=10)

        self.decompressed_img_label = tk.Label(self)
        self.decompressed_img_label.pack(side="right", padx=10)

        self.status_label = tk.Label(self, text="")
        self.status_label.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path and self.is_valid_image(file_path):
            try:
                img = Image.open(file_path)
                img = resize_image(img.convert("RGBA"))
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")
                self.original_image_data = img_bytes.getvalue()

                tk_img = ImageTk.PhotoImage(img)
                self.original_img_label.config(image=tk_img)
                self.original_img_label.image = tk_img

                self.img_hash = get_image_hash(self.original_image_data)
                self.compress_btn.config(state="normal")
                self.status_label.config(text="Image loaded successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load the image: {e}")
        else:
            messagebox.showerror("Error", "Invalid file selected.")

    def is_valid_image(file_path):
        try:
            Image.open(file_path)
            return True
        except IOError:
            return False

    def compress_image(self):
        self.compress_btn.config(state="disabled")
        self.status_label.config(text="Compressing image...")
        Thread(target=self.async_compress_image, daemon=True).start()

    def async_compress_image(self):
        compressed_data = compress_image_data(self.original_image_data)
        asyncio.run(insert_compressed_image(self.img_hash, compressed_data))
        self.status_label.config(text="Image compressed and stored successfully.")
        self.compress_btn.config(state="normal")

    def decompress_and_show(self):
        Thread(target=self.async_decompress_and_show, daemon=True).start()

    def async_decompress_and_show(self):
        decompressed_data = asyncio.run(get_compressed_image(self.img_hash))
        if decompressed_data:
            img = Image.open(io.BytesIO(decompress_image_data(decompressed_data)))
            tk_img = ImageTk.PhotoImage(img)
            self.decompressed_img_label.config(image=tk_img)
            self.decompressed_img_label.image = tk_img
            self.status_label.config(text="Image decompressed successfully.")


if __name__ == "__main__":
    app = ImageCompressorApp()
    app.mainloop()
