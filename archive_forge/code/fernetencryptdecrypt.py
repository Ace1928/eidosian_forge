from cryptography.fernet import Fernet

# File to encrypt
file_to_encrypt = "sensitive_data.txt"

# Generate a key
key = Fernet.generate_key()

# Create a Fernet cipher using the key
cipher = Fernet(key)

# Read the file contents
with open(file_to_encrypt, "rb") as file:
    data = file.read()

# Encrypt the data
encrypted_data = cipher.encrypt(data)

# Write the encrypted data to a new file
with open("encrypted_file.txt", "wb") as file:
    file.write(encrypted_data)

print("File encrypted successfully.")

# Decrypt the file
with open("encrypted_file.txt", "rb") as file:
    encrypted_data = file.read()

decrypted_data = cipher.decrypt(encrypted_data)

# Write the decrypted data to a new file
with open("decrypted_file.txt", "wb") as file:
    file.write(decrypted_data)

print("File decrypted successfully.")
