import glob

# Directory to search in
directory = "path/to/directory"

# Search for files with a specific extension
extension = "*.txt"
files = glob.glob(directory + "/" + extension)

# Print the matching files
for file in files:
    print(file)
