import os
import shutil

# Source directory containing the files to copy
source_directory = "path/to/source/directory"

# Destination directory to copy the files to
destination_directory = "path/to/destination/directory"

# Iterate over the files in the source directory
for filename in os.listdir(source_directory):
    source_path = os.path.join(source_directory, filename)
    destination_path = os.path.join(destination_directory, filename)

    # Copy the file from source to destination
    shutil.copy2(source_path, destination_path)
    print(f"Copied: {filename}")

print("File copying completed.")
