# image_processing.py
import configparser
import os
from PIL import Image, ImageEnhance, ExifTags
import io
import zstd  # Updated to use zstd for compression
import hashlib
from cryptography.fernet import Fernet
from typing import Optional
import logging
from logging_config import configure_logging


# Ensure config and read values
def ensure_config(section, default_values):
    config_path = "config.ini"
    config = configparser.ConfigParser()
    config.read(config_path)
    if not config.has_section(section):
        config.add_section(section)
        for key, value in default_values.items():
            config.set(section, key, value)
        with open(config_path, "w") as configfile:
            config.write(configfile)
    return config


config = ensure_config(
    "ImageProcessing", {"MaxSize": "800,600", "EnhancementFactor": "1.5"}
)
MAX_SIZE = tuple(map(int, config.get("ImageProcessing", "MaxSize").split(",")))
ENHANCEMENT_FACTOR = float(config.get("ImageProcessing", "EnhancementFactor"))

configure_logging()


class BaseImageOperation:
    """
    Base class for image operations.
    """

    def process(self, image_data: bytes) -> bytes:
        """
        Process image data. This method should be overridden by subclasses.

        Args:
            image_data (bytes): The raw image data.

        Returns:
            bytes: The processed image data.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class ContrastEnhancement(BaseImageOperation):
    """
    Enhances the contrast of an image.
    """

    def process(
        self, image_data: bytes, enhancement_factor: float = ENHANCEMENT_FACTOR
    ) -> bytes:
        try:
            image = Image.open(io.BytesIO(image_data))
            enhancer = ImageEnhance.Contrast(image)
            enhanced_image = enhancer.enhance(enhancement_factor)
            with io.BytesIO() as output:
                enhanced_image.save(output, format=image.format)
                return output.getvalue()
        except Exception as e:
            logging.error(f"Error enhancing image contrast: {e}")
            raise


def resize_image(image: Image.Image, max_size: tuple = MAX_SIZE) -> Image.Image:
    """
    Resizes an image to fit within a maximum size while maintaining aspect ratio.

    Args:
        image (Image.Image): The original image.
        max_size (tuple): A tuple of (max_width, max_height).

    Returns:
        Image.Image: The resized image.
    """
    try:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        logging.debug(f"Image resized to max size {max_size}.")
        return image
    except Exception as e:
        logging.error(f"Error resizing image: {e}")
        raise


def get_image_hash(image_data: bytes) -> str:
    """
    Hashes the image data using SHA-512 to ensure uniqueness and increased security.

    Args:
        image_data (bytes): The raw image data.

    Returns:
        str: The hexadecimal hash of the image.
    """
    try:
        sha512_hash = hashlib.sha512()
        sha512_hash.update(image_data)
        logging.debug("Image hash successfully generated using SHA-512.")
        return sha512_hash.hexdigest()
    except Exception as e:
        logging.error(f"Error generating image hash with SHA-512: {e}")
        raise


def generate_checksum(data: bytes) -> str:
    """Generate a SHA-256 checksum for the given data."""
    return hashlib.sha256(data).hexdigest()


def verify_checksum(data: bytes, expected_checksum: str) -> bool:
    """Verify the data integrity by comparing the generated checksum with the expected one."""
    return generate_checksum(data) == expected_checksum


def compress_and_encrypt_image_data(
    image_data: bytes, image_format: str, cipher_suite: Fernet
) -> bytes:
    try:
        checksum = generate_checksum(image_data)
        formatted_data = (
            image_format.encode() + b"\x00" + checksum.encode() + b"\x00" + image_data
        )
        compressed_data = zstd.compress(formatted_data)
        encrypted_data = cipher_suite.encrypt(compressed_data)
        logging.debug("Compression and encryption successful with checksum.")
        return encrypted_data
    except Exception as e:
        logging.error("Compression and encryption failed: {}".format(e))
        raise


def decrypt_and_decompress_image_data(
    encrypted_data: bytes, cipher_suite: Fernet
) -> tuple:
    try:
        decrypted_data = cipher_suite.decrypt(encrypted_data)
        decompressed_data = zstd.decompress(decrypted_data)
        image_format, checksum, image_data = decompressed_data.split(b"\x00", 2)

        if not verify_checksum(image_data, checksum.decode()):
            raise ValueError(
                "Checksum verification failed after decryption and decompression."
            )

        logging.debug(
            "Decryption and decompression successful with checksum verification."
        )
        return image_data, image_format.decode()
    except Exception as e:
        logging.error("Decryption and decompression failed: {}".format(e))
        raise


def validate_image(image: Image.Image) -> bool:
    """
    Validates the image format and size.

    Args:
        image (Image.Image): The image to validate.

    Returns:
        bool: True if the image is valid, False otherwise.
    """
    try:
        valid_formats = ["JPEG", "PNG", "BMP", "GIF"]  # Expanded list
        is_valid = (
            image.format in valid_formats
            and image.width <= 4000
            and image.height <= 4000
        )
        logging.debug(f"Image validation result: {is_valid}.")
        return is_valid
    except Exception as e:
        logging.error(f"Error validating image: {e}")
        raise


def is_valid_image(image_data: bytes) -> bool:
    try:
        image = Image.open(io.BytesIO(image_data))
        if (
            image.format in ["JPEG", "PNG", "BMP", "GIF"]
            and image.width <= 4000
            and image.height <= 4000
        ):
            logging.debug("Image validation successful.")
            return True
        else:
            logging.warning(
                "Image failed validation: Unsupported format or dimensions."
            )
            return False
    except Exception as e:
        logging.error("Image validation error: {}".format(e))
        return False


def read_image_metadata(image_data: bytes) -> dict:
    """
    Reads EXIF metadata from an image.

    Args:
        image_data (bytes): The raw image data.

    Returns:
        dict: A dictionary containing EXIF metadata, if available.
    """
    with Image.open(io.BytesIO(image_data)) as image:
        exif_data = {}
        if hasattr(image, "_getexif"):
            exif = image._getexif()
            if exif is not None:
                for tag, value in exif.items():
                    decoded = ExifTags.TAGS.get(tag, tag)
                    exif_data[decoded] = value
        return exif_data
