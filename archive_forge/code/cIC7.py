import io
import os
import logging
from PIL import Image, ImageEnhance, ExifTags
import hashlib
import zstd
from cryptography.fernet import Fernet
from typing import Tuple, Dict, Any
from config_manager import ConfigManager  # Assuming proper path
from logging_config import configure_logging

# Initialize logging
configure_logging()

# Load configurations using ConfigManager
config_manager = ConfigManager()
config_path = os.path.join(os.path.dirname(__file__), "config.ini")
config_manager.load_config(config_path, "ImageProcessing")
MAX_SIZE = tuple(
    map(
        int,
        config_manager.get("ImageProcessing", "MaxSize", fallback="800,600").split(","),
    )
)
ENHANCEMENT_FACTOR = float(
    config_manager.get("ImageProcessing", "EnhancementFactor", fallback="1.5")
)


class ImageOperationError(Exception):
    """Custom exception for image operation errors."""


def ensure_image_format(image_data: bytes) -> Image.Image:
    """
    Ensures the given image data can be opened and returns the Image object.

    Args:
        image_data (bytes): The raw image data.

    Returns:
        Image.Image: The opened image.

    Raises:
        ImageOperationError: If the image cannot be opened.
    """
    try:
        return Image.open(io.BytesIO(image_data))
    except Exception as e:
        logging.error(f"Failed to open image: {e}")
        raise ImageOperationError(f"Failed to open image: {e}")


def enhance_contrast(
    image: Image.Image, enhancement_factor: float = ENHANCEMENT_FACTOR
) -> bytes:
    """
    Enhances the contrast of an image.

    Args:
        image (Image.Image): The image to enhance.
        enhancement_factor (float): The factor by which to enhance the image's contrast.

    Returns:
        bytes: The enhanced image data.

    Raises:
        ImageOperationError: If contrast enhancement fails.
    """
    try:
        enhancer = ImageEnhance.Contrast(image)
        enhanced_image = enhancer.enhance(enhancement_factor)
        with io.BytesIO() as output:
            enhanced_image.save(output, format=image.format)
            return output.getvalue()
    except Exception as e:
        logging.error(f"Error enhancing image contrast: {e}")
        raise ImageOperationError(f"Error enhancing image contrast: {e}")


def resize_image(
    image: Image.Image, max_size: Tuple[int, int] = MAX_SIZE
) -> Image.Image:
    """
    Resizes an image to fit within a maximum size while maintaining aspect ratio.

    Args:
        image (Image.Image): The original image.
        max_size (Tuple[int, int]): A tuple of (max_width, max_height).

    Returns:
        Image.Image: The resized image.

    Raises:
        ImageOperationError: If resizing the image fails.
    """
    try:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        logging.debug(f"Image resized to max size {max_size}.")
        return image
    except Exception as e:
        logging.error(f"Error resizing image: {e}")
        raise ImageOperationError(f"Error resizing image: {e}")


def get_image_hash(image_data: bytes) -> str:
    """
    Generates a SHA-512 hash of the image data.

    Args:
        image_data (bytes): The raw image data.

    Returns:
        str: The hexadecimal hash of the image.
    """
    sha512_hash = hashlib.sha512()
    sha512_hash.update(image_data)
    logging.debug("Image hash successfully generated using SHA-512.")
    return sha512_hash.hexdigest()


def compress(image_data: bytes, image_format: str) -> bytes:
    """
    Compresses image data along with its format and a checksum for integrity verification.

    Args:
        image_data (bytes): The raw image data.
        image_format (str): The format of the image.

    Returns:
        bytes: The compressed image data.
    """
    checksum = hashlib.sha256(image_data).hexdigest()
    formatted_data = f"{image_format}\x00{checksum}\x00".encode() + image_data
    compressed_data = zstd.compress(formatted_data)
    logging.debug("Image data compressed successfully.")
    return compressed_data


def encrypt(data: bytes, cipher_suite: Fernet) -> bytes:
    """
    Encrypts data using the provided cipher suite.

    Args:
        data (bytes): The data to encrypt.
        cipher_suite (Fernet): The cipher suite for encryption.

    Returns:
        bytes: The encrypted data.
    """
    encrypted_data = cipher_suite.encrypt(data)
    logging.debug("Data encrypted successfully.")
    return encrypted_data


def decrypt(encrypted_data: bytes, cipher_suite: Fernet) -> bytes:
    """
    Decrypts data using the provided cipher suite.

    Args:
        encrypted_data (bytes): The encrypted data.
        cipher_suite (Fernet): The cipher suite for decryption.

    Returns:
        bytes: The decrypted data.
    """
    decrypted_data = cipher_suite.decrypt(encrypted_data)
    logging.debug("Data decrypted successfully.")
    return decrypted_data


def decompress(data: bytes) -> Tuple[bytes, str]:
    """
    Decompresses data and extracts the image format and raw image data.

    Args:
        data (bytes): The compressed data.

    Returns:
        Tuple[bytes, str]: The raw image data and its format.
    """
    decompressed_data = zstd.decompress(data)
    image_format, checksum, image_data = decompressed_data.split(b"\x00", 2)
    logging.debug("Data decompressed successfully.")
    return image_data, image_format.decode()


def generate_checksum(image_data: bytes) -> str:
    """
    Generates a SHA-256 checksum for the given image data.

    Args:
        image_data (bytes): The raw image data.

    Returns:
        str: The hexadecimal checksum of the image.
    """
    checksum = hashlib.sha256(image_data).hexdigest()
    logging.debug("Checksum generated successfully.")
    return checksum


def verify_checksum(image_data: bytes, expected_checksum: str) -> bool:
    """
    Verifies the integrity of the image data against the expected checksum.

    Args:
        image_data (bytes): The raw image data.
        expected_checksum (str): The expected checksum for verification.

    Returns:
        bool: True if the checksum matches, False otherwise.
    """
    actual_checksum = hashlib.sha256(image_data).hexdigest()
    is_valid = actual_checksum == expected_checksum
    logging.debug(f"Checksum verification result: {is_valid}.")
    return is_valid


def validate_image(image: Image.Image) -> bool:
    """
    Validates the image format and size.

    Args:
        image (Image.Image): The image to validate.

    Returns:
        bool: True if the image is valid, False otherwise.

    Raises:
        ImageOperationError: If image validation fails.
    """
    try:
        valid_formats = ["JPEG", "PNG", "BMP", "GIF"]
        is_valid = (
            image.format in valid_formats
            and image.width <= 4000
            and image.height <= 4000
        )
        logging.debug(f"Image validation result: {is_valid}.")
        return is_valid
    except Exception as e:
        logging.error(f"Error validating image: {e}")
        raise ImageOperationError(f"Error validating image: {e}")


def read_image_metadata(image_data: bytes) -> Dict[str, Any]:
    """
    Reads EXIF metadata from an image.

    Args:
        image_data (bytes): The raw image data.

    Returns:
        Dict[str, Any]: A dictionary containing EXIF metadata, if available.

    Raises:
        ImageOperationError: If reading metadata fails.
    """
    try:
        with Image.open(io.BytesIO(image_data)) as image:
            exif_data = {}
            if hasattr(image, "_getexif"):
                exif = image._getexif()
                if exif is not None:
                    for tag, value in exif.items():
                        decoded = ExifTags.TAGS.get(tag, tag)
                        exif_data[decoded] = value
            logging.debug("Image metadata read successfully.")
            return exif_data
    except Exception as e:
        logging.error(f"Error reading image metadata: {e}")
        raise ImageOperationError(f"Error reading image metadata: {e}")
