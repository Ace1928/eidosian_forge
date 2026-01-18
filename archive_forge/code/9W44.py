import asyncio
import io
import tkinter as tk
import threading
from threading import Thread
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk
from cryptography.fernet import Fernet

from core_services import LoggingManager, EncryptionManager, ConfigManager
from database import (
    init_db,
    insert_compressed_image,
    retrieve_compressed_image,
    get_images_metadata,
)
from image_processing import (
    compress,
    encrypt,
    decrypt,
    decompress,
    get_image_hash,
    validate_image,
    ensure_image_format,
)

LoggingManager.configure_logging()


class ImageCompressorApp(tk.Tk):
    """
    A GUI application for compressing and decompressing images securely.

    Attributes:
        title (str): The title of the main application window.
        geometry (str): The size and position of the main application window.
        original_image_data (bytes, optional): The original data of the loaded image.
        img_hash (str, optional): The hash of the currently loaded image.
        cipher_suite (Fernet): The cryptography suite for encryption and decryption.
        current_page (int): The current page number in the image browsing view.
        items_per_page (int): The number of items to display per page in the browsing view.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # Initialize the asyncio event loop for background tasks
        self.asyncio_loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self.start_asyncio_loop, daemon=True)
        self.loop_thread.start()
        self.title("Image Compressor")
        self.geometry("800x600")  # Adjust the window size as needed
        self.original_image_data: bytes | None = None
        self.img_hash: str | None = None
        self.cipher_suite: Fernet = Fernet(EncryptionManager.get_valid_encryption_key())
        self.current_page: int = 0
        self.items_per_page: int = 10
        self.setup_ui()
        self.initialize_background_tasks()

    def start_asyncio_loop(self):
        """Runs the asyncio event loop."""
        asyncio.set_event_loop(self.asyncio_loop)
        self.asyncio_loop.run_forever()

    def setup_ui(self) -> None:
        """
        Sets up the user interface components of the application.
        """
        self.init_widgets()

    def initialize_background_tasks(self) -> None:
        """
        Initializes background tasks necessary for the application, such as database initialization.
        """
        Thread(target=lambda: asyncio.run(self.async_init_db()), daemon=True).start()

    async def async_init_db(self) -> None:
        """
        Asynchronously initializes the database to avoid blocking the GUI.
        """
        await init_db()
        LoggingManager.info("Database initialized successfully.")

    def init_widgets(self) -> None:
        self.load_btn = tk.Button(self, text="Load Image", command=self.load_image)
        self.load_btn.pack()

        self.compress_btn = tk.Button(
            self,
            text="Compress & Store Image",
            state="disabled",
            command=self.compress_and_store_image,
        )
        self.compress_btn.pack()

        self.decompress_btn = tk.Button(
            self, text="Decompress & Show Image", command=self.decompress_and_show
        )
        self.decompress_btn.pack()

        self.browse_db_btn = tk.Button(
            self, text="Browse Stored Images", command=self.browse_stored_images
        )
        self.browse_db_btn.pack()

        self.original_img_label = tk.Label(self)
        self.original_img_label.pack(side="left", padx=10)

        self.decompressed_img_label = tk.Label(self)
        self.decompressed_img_label.pack(side="right", padx=10)

        self.status_label = tk.Label(self, text="")
        self.status_label.pack()

        self.progress_bar = ttk.Progressbar(
            self, orient="horizontal", mode="indeterminate"
        )
        self.progress_bar.pack()

        # Pagination controls
        self.pagination_frame = tk.Frame(self)
        self.prev_button = tk.Button(
            self.pagination_frame, text="Previous", command=self.previous_page
        )
        self.next_button = tk.Button(
            self.pagination_frame, text="Next", command=self.next_page
        )
        self.prev_button.pack(side=tk.LEFT, padx=10)
        self.next_button.pack(side=tk.RIGHT, padx=10)
        self.pagination_frame.pack(side=tk.BOTTOM, pady=10)

        # Image display area
        self.image_display_frame = tk.Frame(self)
        self.image_display_frame.pack(fill=tk.BOTH, expand=True)
        self.image_labels = [
            tk.Label(self.image_display_frame) for _ in range(self.items_per_page)
        ]
        for label in self.image_labels:
            label.pack()

    def update_image_display(self, metadata_list):
        # Clear existing images
        for label in self.image_labels:
            label.config(image="")
            label.image = None

        # Display new images
        for metadata, label in zip(metadata_list, self.image_labels):
            try:
                img_data = metadata[0]  # Assuming metadata is a tuple (image_data, ...)
                img = Image.open(io.BytesIO(img_data))
                img.thumbnail((100, 100), Image.ANTIALIAS)  # Example thumbnail size
                tk_img = ImageTk.PhotoImage(img)
                label.config(image=tk_img)
                label.image = tk_img  # Keep a reference
            except Exception as e:
                LoggingManager.error(f"Error displaying image: {e}")

    async def async_browse_stored_images_coroutine(self):
        offset = self.current_page * self.items_per_page
        limit = self.items_per_page
        try:
            metadata_list = await get_images_metadata(offset, limit)
            self.after(0, self.update_image_display, metadata_list)
        except Exception as e:
            LoggingManager.error(f"Failed to browse stored images: {e}")
            messagebox.showerror("Error", f"Failed to browse stored images: {e}")

    def async_browse_stored_images(self) -> None:
        asyncio.run_coroutine_threadsafe(
            self.async_browse_stored_images_coroutine(), asyncio.get_event_loop()
        )

    def next_page(self) -> None:
        self.current_page += 1
        self.async_browse_stored_images()

    def previous_page(self) -> None:
        if self.current_page > 0:
            self.current_page -= 1
            self.async_browse_stored_images()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            with open(file_path, "rb") as f:
                image_data = f.read()
            image = ensure_image_format(image_data)
            if image and validate_image(image):
                self.original_image_data = image_data
                self.img_hash = get_image_hash(self.original_image_data)
                tk_img = ImageTk.PhotoImage(image)
                self.original_img_label.config(image=tk_img)
                self.original_img_label.image = tk_img  # Keep a reference
                self.compress_btn.config(state="normal")
                self.update_status("Image loaded successfully.")
            else:
                messagebox.showerror("Error", "Invalid image format or size.")

    def compress_and_store_image(self) -> None:
        """
        Initiates the process to compress and store the currently loaded image.
        """
        self.compress_btn.config(state="disabled")
        self.update_status("Compressing image...")
        self.progress_bar.start()
        # Dispatch the coroutine to the asyncio event loop
        future = asyncio.run_coroutine_threadsafe(
            self.async_compress_and_store_image(), self.asyncio_loop
        )
        future.add_done_callback(
            lambda x: self.after(0, self.on_compress_and_store_image_done, x)
        )

    def on_compress_and_store_image_done(self, future):
        """
        Callback function to be called when async_compress_and_store_image coroutine is done.
        """
        try:
            future.result()  # This retrieves the result of the coroutine or re-raises any exception thrown
            self.update_status("Image compressed, encrypted, and stored successfully.")
        except Exception as e:
            LoggingManager.error(f"Failed to compress, encrypt, and store image: {e}")
            self.update_status(
                "Failed to compress, encrypt, and store image. Please try again.",
                error=True,
            )
        finally:
            self.progress_bar.stop()
            self.compress_btn.config(state="normal")

    async def async_compress_and_store_image(self):
        try:
            compressed_data = await compress(self.original_image_data, "PNG")
            encrypted_data = await encrypt(compressed_data)
            await insert_compressed_image(self.img_hash, "PNG", encrypted_data)
            self.after(
                0,
                lambda: self.update_status(
                    "Image compressed, encrypted, and stored successfully."
                ),
            )
        except Exception as e:
            LoggingManager.error(f"Failed to compress, encrypt, and store image: {e}")
            self.after(
                0,
                lambda: self.update_status(
                    "Failed to compress, encrypt, and store image. Please try again.",
                    error=True,
                ),
            )

    def decompress_and_show(self) -> None:
        """
        Initiates the process to decompress and display the stored image.
        """
        self.progress_bar.start()
        future = asyncio.run_coroutine_threadsafe(
            self.async_decompress_and_show(), self.asyncio_loop
        )
        future.add_done_callback(
            lambda x: self.after(0, self.on_decompress_and_show_done, x)
        )

    def on_decompress_and_show_done(self, future):
        """
        Callback function to be called when async_decompress_and_show coroutine is done.
        """
        try:
            future.result()  # This retrieves the result of the coroutine or re-raises any exception thrown
            self.update_status("Image decompressed successfully.")
        except Exception as e:
            LoggingManager.error(f"Failed to decompress image: {e}")
            self.update_status(
                "Failed to decompress image. Please try again.", error=True
            )
        finally:
            self.progress_bar.stop()

    async def async_decompress_and_show(self):
        try:
            encrypted_data = await retrieve_compressed_image(self.img_hash)
            decrypted_data = await decrypt(encrypted_data)
            decompressed_data = await decompress(decrypted_data)
            img = Image.open(io.BytesIO(decompressed_data))
            tk_img = ImageTk.PhotoImage(img)
            self.decompressed_img_label.config(image=tk_img)
            self.decompressed_img_label.image = tk_img  # Keep a reference
            self.update_status("Image decompressed successfully.")
        except Exception as e:
            LoggingManager.error(f"Failed to decompress image: {e}")
            self.update_status(
                "Failed to decompress image. Please try again.", error=True
            )

    def browse_stored_images(self) -> None:
        """
        Initiates the browsing of stored images starting from the first page.
        """
        self.current_page = 0  # Reset to the first page
        self.async_browse_stored_images()

    def update_status(self, message: str, error: bool = False) -> None:
        """
        Updates the status label with a message, optionally marking it as an error.

        Args:
            message (str): The message to display.
            error (bool): If True, the message is treated as an error.
        """
        self.status_label.config(text=message, fg="red" if error else "black")


if __name__ == "__main__":
    app = ImageCompressorApp()
    app.mainloop()
