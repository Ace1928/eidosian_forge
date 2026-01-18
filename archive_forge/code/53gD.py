import os

# File to modify permissions
file_path = "path/to/file"

# Set file permissions
os.chmod(file_path, 0o755)

print("File permissions modified successfully.")
