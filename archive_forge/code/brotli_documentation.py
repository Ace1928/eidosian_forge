import _brotli
Compress a byte string.

    Args:
      string (bytes): The input data.
      mode (int, optional): The compression mode can be MODE_GENERIC (default),
        MODE_TEXT (for UTF-8 format text input) or MODE_FONT (for WOFF 2.0).
      quality (int, optional): Controls the compression-speed vs compression-
        density tradeoff. The higher the quality, the slower the compression.
        Range is 0 to 11. Defaults to 11.
      lgwin (int, optional): Base 2 logarithm of the sliding window size. Range
        is 10 to 24. Defaults to 22.
      lgblock (int, optional): Base 2 logarithm of the maximum input block size.
        Range is 16 to 24. If set to 0, the value will be set based on the
        quality. Defaults to 0.

    Returns:
      The compressed byte string.

    Raises:
      brotli.error: If arguments are invalid, or compressor fails.
    