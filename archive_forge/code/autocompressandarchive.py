import os
import zipfile

# Directory containing the files to compress
directory = "/path/to/directory"

# Name of the output ZIP archive
zip_filename = "archive.zip"

# Create a new ZIP archive
with zipfile.ZipFile(zip_filename, "w") as zip_file:
    # Iterate over the files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Add each file to the ZIP archive
        zip_file.write(file_path, filename)

print(f"Files compressed into {zip_filename}")
