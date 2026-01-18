import os
import time

# Directory containing the files
directory = "path/to/directory"

# Age threshold in seconds (e.g., 30 days)
age_threshold = 30 * 24 * 60 * 60

# Get the current timestamp
current_time = time.time()

# Iterate over the files in the directory
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)

    # Check if the file is older than the age threshold
    if (
        os.path.isfile(file_path)
        and current_time - os.path.getmtime(file_path) > age_threshold
    ):
        # Delete the file
        os.remove(file_path)
        print(f"Deleted: {filename}")

print("File deletion completed.")
