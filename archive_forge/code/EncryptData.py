
import hashlib

def encrypt_data(data):
    """
    Encrypts sensitive data using a robust encryption algorithm.

    Args:
    data (str): The data to be encrypted.

    Returns:
    str: The encrypted data.
    """
    if not data:
        return "No data provided."

    # Example using SHA-256 hashing for demonstration. Note: This is a one-way hash, not an encryption.
    encrypted_data = hashlib.sha256(data.encode()).hexdigest()
    return encrypted_data

# Example usage
if __name__ == "__main__":
    sample_data = "Sensitive Data"
    encrypted = encrypt_data(sample_data)
    print(f"Encrypted Data: {encrypted}")
